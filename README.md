# Anomaly Detection API

## Setup

1. **Create a Virtual Environment**


    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

2. **Install Dependencies**

    Install all necessary dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

3. **Configure the `.env` File**

    Create a `.env` file in the root directory of the project and define the necessary environment variables (e.g., database URL, JWT secret key):

    ```bash
    touch .env
    ```

    Add the following values to `.env` (replace with your actual values):

    ```
    SQLALCHEMY_DATABASE_URI=postgresql://username:password@localhost:port/database_name
    JWT_SECRET_KEY=your_jwt_secret_key
    ```

    **Note**: Make sure to configure your database URI correctly (PostgreSQL, SQLite, etc.) and provide a secure JWT secret key.

---

## Create Database and Admin

Before running the app, you'll need to set up the database and create an admin user:

1. **Create the Database**

    Ensure that your database (as specified in the `.env` file) is created. For PostgreSQL, you can run:

    ```bash
    psql -U your_username -d your_database_name
    ```

    You can also create the database via a database client (e.g., pgAdmin, DBeaver) or through the PostgreSQL CLI.

2. **Initialize the Database**

    Once the database is set up, run the following script to create the necessary tables:

    ```bash
    python create_admin.py
    ```

    This script will:
    - Create the tables for your `User` model in the database.
    - Check if an admin user exists. If not, it will create a user with the username `admin` and the password `admin123`.

    **Note**: Ensure that your `SQLALCHEMY_DATABASE_URI` in `.env` points to a valid database.

---

## Running the Application

Once the database is initialized and the admin user is created, you can run the application with:

```bash
python app_v3.py
