from app_v3 import app, db, bcrypt, User

with app.app_context():
    db.create_all()

def create_admin():
    with app.app_context():
        # cek admin 
        if not User.query.filter_by(username='admin').first():
            hashed_password = bcrypt.generate_password_hash('admin123').decode('utf-8')
            admin = User(
                username='admin',
                password=hashed_password,
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()
            print("Admin user created successfully")
        else:
            print("Admin user already exists")

if __name__ == "__main__":
    create_admin()
