import json
from datetime import datetime
import google.generativeai as genai

from flask import Flask, request, jsonify, render_template, make_response, current_app
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate
from config import Config
from models import db, User, Entry
from pathlib import Path
from predict_model import predict_emotion

BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"

def create_app():
    app = Flask(__name__, template_folder=str(FRONTEND_DIR), static_folder=str(FRONTEND_DIR))
    app.config.from_object(Config)

    if app.config.get("GOOGLE_API_KEY"):
        genai.configure(api_key=app.config["GOOGLE_API_KEY"])

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # DB ve migrate
    db.init_app(app)
    Migrate(app, db)

    # app.py içi

    def ai_journal_coach(answers_dict, entry_text, emotion, user_lang="auto"):
        """
        answers_dict: signup anketi (dict)
        entry_text: güncel entry
        emotion: predict_model’dan gelen duygusal etiket (ör. "sad", "happy"...)
        user_lang: "auto" → entry hangi dilse o dilde yanıtla
        """

        # Anketi bullet list olarak kısaca özetleyelim (çok uzunsa trime edebiliriz)
        def summarize_answers(ans):
            if not isinstance(ans, dict):
                return ""
            items = []
            for k, v in ans.items():
                v = (v or "").strip()
                if not v:
                    continue
                # çok uzunsa kısalt
                if len(v) > 200:
                    v = v[:200] + "..."
                items.append(f"- {k}: {v}")
            return "\n".join(items[:10])  # güvenli üst sınır

        answers_summary = summarize_answers(answers_dict)

        style = (
            "Write 2-3 short, supportive sentences with 2-4 cute, relevant emojis. "
            "End with one tiny actionable tip. Avoid clinical wording. "
            "If the entry language is Turkish, reply in Turkish; otherwise reply in the entry language."
            if user_lang == "auto" else
            f"Reply in {user_lang}. Use 2-4 cute emojis, 2-3 short sentences, one tiny tip."
        )

        prompt = f"""
    You are a gentle journaling buddy.
    User profile quick notes (from onboarding survey):
    {answers_summary if answers_summary else "(no survey answers)"} 

    Today's entry (verbatim):
    \"\"\"{entry_text}\"\"\"

    Detected emotion label (not shown to user): {emotion}

    Task:
    - Give a short, warm reflection and encouragement.
    - 2-4 tasteful, context-matching emojis.
    - 2-3 sentences total, then one micro-tip on a new line starting with "Tip:".
    - No therapy/diagnosis, no disclaimers.

    {style}
    """.strip()

        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            text = (resp.text or "").strip()
            # Güvenlik: çok uzun gelirse kısalt
            return text[:1000] if text else None
        except Exception as e:
            print("Gemini error:", e)
            return None

    # ---------- PAGES ----------
    @app.get("/")
    def index():
        print("hello world")
        resp = make_response(render_template("loginSignUpPage.html"))
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    @app.get("/journal")
    def journal():
        resp = make_response(render_template("journalPage.html"))
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    @app.get("/profile")
    def profile():
        resp = make_response(render_template("profilePage.html"))
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    # ---------- HEALTH ----------
    @app.get("/health")
    def health():
        db.session.execute(db.text("SELECT 1"))
        return "ok", 200

    # ---------- AUTH ----------
    @app.post("/api/signup")
    def signup():
        data = request.get_json()
        name, email, password, answers = (
            data.get("name"),
            data.get("email"),
            data.get("password"),
            data.get("answers")
        )
        if isinstance(answers, dict):
            answers = json.dumps(answers)
        # Alanların boş olup olmadığını kontrol et
        if not all([name, email, password, answers]):
            if (answers is None):
                print("cevaplar alınamadı")
            return jsonify({"error": "Missing fields"}), 400

        # E-postanın zaten kayıtlı olup olmadığını kontrol et
        if User.query.filter_by(email=email).first():
            return jsonify({"error": "Email already registered"}), 409

        # Şifreyi hash'le ve kullanıcıyı veritabanına ekle
        hashed_password = generate_password_hash(password)
        user = User(
            name=name,
            email=email,
            password_hash=hashed_password,
            answers=answers
        )

        db.session.add(user)
        db.session.commit()

        return jsonify({"success": True, "message": "User created"}), 201

    @app.post("/api/login")
    def login():
        data = request.form if request.form else request.get_json(silent=True) or {}
        email, password = data.get("email"), data.get("password")
        if not all([email, password]):
            return jsonify({"error": "Missing fields"}), 400
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({"error": "Invalid credentials"}), 401
        return jsonify({"message": "Login ok", "user": {"id": user.id, "name": user.name, "email": user.email}}), 200

    # ---------- USERS ----------
    @app.get("/api/users/<int:user_id>")
    def get_user(user_id):
        u = User.query.get(user_id)
        if not u:
            return jsonify({"error": "not found"}), 404
        return jsonify({
            "id": u.id,
            "name": u.name,
            "email": u.email,
            "member_since": u.created_at.isoformat() if u.created_at else None
        }), 200

    @app.put("/api/users/<int:user_id>/password")
    def change_password(user_id):
        data = request.get_json(silent=True) or {}
        current_password = (data.get("current_password") or "").strip()
        new_password = (data.get("new_password") or "").strip()

        if not current_password or not new_password:
            return jsonify({"error": "current_password and new_password required"}), 400

        u = User.query.get(user_id)
        if not u:
            return jsonify({"error": "not found"}), 404

        # Mevcut şifre doğru mu?
        if not check_password_hash(u.password_hash, current_password):
            return jsonify({"error": "current password is incorrect"}), 401

        u.password_hash = generate_password_hash(new_password)
        db.session.commit()
        return jsonify({"message": "password updated"}), 200

    @app.delete("/api/users/<int:user_id>")
    def delete_account(user_id):
        data = request.get_json(silent=True) or {}
        current_password = (data.get("current_password") or "").strip()
        if not current_password:
            return jsonify({"error": "current_password required"}), 400

        u = User.query.get(user_id)
        if not u:
            return jsonify({"error": "not found"}), 404

        if not check_password_hash(u.password_hash, current_password):
            return jsonify({"error": "current password is incorrect"}), 401

        # Kullanıcının tüm entry'lerini sil → sonra kullanıcıyı sil
        db.session.query(Entry).filter_by(user_id=user_id).delete(synchronize_session=False)
        db.session.delete(u)
        db.session.commit()
        return "", 204

    # ---------- ENTRIES ----------

    @app.post("/api/entries")
    def create_entry():
        data = request.get_json(silent=True) or {}
        user_id = data.get("user_id")
        content = (data.get("content") or "").strip()

        if not user_id or not content:
            return jsonify({"error": "user_id and content required"}), 400

        # Duygu tahmini
        emotion = predict_emotion(content)

        # Konsola yazdır
        print(f"[DUYGU TAHMİNİ] User {user_id}: '{content}' --> {emotion}")

        # Kullanıcı + anket
        u = User.query.get(user_id)
        answers_dict = {}
        if u and u.answers:
            try:
                answers_dict = json.loads(u.answers)
            except Exception:
                answers_dict = {}

        # Veritabanına kaydet (emotion olmadan)
        e = Entry(user_id=user_id, content=content, emotion=emotion)
        db.session.add(e)
        db.session.commit()

        # Gemini’den koç yanıtı iste (senkron; istersen try/except içinde)
        ai_reply = None
        if current_app.config.get("GOOGLE_API_KEY"):
            ai_reply = ai_journal_coach(answers_dict, content, emotion, user_lang="auto")

        # DB’de sakla (başarısız olursa boş kalır)
        if ai_reply:
            e.ai_reply = ai_reply
            db.session.commit()
        # Response içinde emotion döndür
        return jsonify({
            "id": e.id,
            "content": e.content,
            "emotion": e.emotion,  # artık kalıcı
            "ai_reply": e.ai_reply,  # kısa koç cevabı
            "created_at": e.created_at.isoformat() if e.created_at else None,
            # sadece varsa gönderelim
            **({"updated_at": e.updated_at.isoformat()} if e.updated_at else {})
        }), 201

    @app.get("/api/entries/<int:user_id>")
    def list_entries(user_id):
        rows = Entry.query.filter_by(user_id=user_id).order_by(Entry.id.desc()).all()
        return jsonify([
            {"id": r.id, "content": r.content,
             "created_at": r.created_at.isoformat() if r.created_at else None,
             "updated_at": r.updated_at.isoformat() if r.updated_at else None }
            for r in rows
        ])

    @app.put("/api/entries/<int:entry_id>")
    def update_entry(entry_id):
        data = request.get_json(silent=True) or {}
        content = (data.get("content") or "").strip()
        if not content:
            return jsonify({"error": "content required"}), 400

        e = Entry.query.get(entry_id)
        if not e:
            return jsonify({"error": "not found"}), 404

        e.content = content
        e.updated_at = datetime.now()
        db.session.commit()
        return jsonify({
            "id": e.id,
            "content": e.content,
            "created_at": e.created_at.isoformat() if e.created_at else None,
            "updated_at": datetime.now().isoformat() if e.updated_at else None,
        }), 200

    # DELETE /api/entries/<id>
    @app.delete("/api/entries/<int:entry_id>")
    def delete_entry(entry_id):
        user_id = request.args.get("user_id", type=int)  # ?user_id=...
        e = Entry.query.get(entry_id)
        if not e:
            return jsonify({"error": "not found"}), 404
        if user_id and e.user_id != user_id:
            return jsonify({"error": "forbidden"}), 403
        db.session.delete(e)
        db.session.commit()
        return "", 204

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, use_reloader=False)
