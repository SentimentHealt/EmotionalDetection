import json
from datetime import datetime

from flask import Flask, request, jsonify, render_template, make_response
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
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # DB ve migrate
    db.init_app(app)
    Migrate(app, db)



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
            if(answers is None):
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

        # Veritabanına kaydet (emotion olmadan)
        e = Entry(user_id=user_id, content=content,emotion=emotion,updated_at=datetime.now())
        db.session.add(e)
        db.session.commit()

        # Response içinde emotion döndür
        return jsonify({
            "id": e.id,
            "content": e.content,
            "emotion": emotion,  # kalıcı değil, sadece response
            "created_at": e.created_at.isoformat() if e.created_at else None
        }), 201

    @app.get("/api/entries/<int:user_id>")
    def list_entries(user_id):
        rows = Entry.query.filter_by(user_id=user_id).order_by(Entry.id.desc()).all()
        return jsonify([
            {"id": r.id, "content": r.content,
             "created_at": r.created_at.isoformat() if r.created_at else None,
             "updated_at": r.updated_at.isoformat() if r.updated_at else None,}
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
