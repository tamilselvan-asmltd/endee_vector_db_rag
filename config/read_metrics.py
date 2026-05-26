import os
import sqlite3
from sqlite3 import Row
from datetime import date

DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "health.db"))

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    email           TEXT NOT NULL UNIQUE,
    display_name    TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS daily_body_metrics (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    measurement_date            TEXT NOT NULL,
    blood_sugar_fasting         REAL CHECK (blood_sugar_fasting IS NULL OR (blood_sugar_fasting >= 20 AND blood_sugar_fasting <= 500)),
    blood_sugar_post_meal       REAL CHECK (blood_sugar_post_meal IS NULL OR (blood_sugar_post_meal >= 20 AND blood_sugar_post_meal <= 500)),
    systolic_bp                 INTEGER CHECK (systolic_bp IS NULL OR (systolic_bp >= 50 AND systolic_bp <= 300)),
    diastolic_bp                INTEGER CHECK (diastolic_bp IS NULL OR (diastolic_bp >= 30 AND diastolic_bp <= 200)),
    heart_rate_bpm              INTEGER CHECK (heart_rate_bpm IS NULL OR (heart_rate_bpm >= 20 AND heart_rate_bpm <= 250)),
    oxygen_saturation_percent   REAL CHECK (oxygen_saturation_percent IS NULL OR (oxygen_saturation_percent >= 50 AND oxygen_saturation_percent <= 100)),
    body_temperature_celsius    REAL CHECK (body_temperature_celsius IS NULL OR (body_temperature_celsius >= 34 AND body_temperature_celsius <= 42)),
    body_weight_kg              REAL CHECK (body_weight_kg IS NULL OR (body_weight_kg >= 20 AND body_weight_kg <= 500)),
    bmi                         REAL CHECK (bmi IS NULL OR (bmi >= 10 AND bmi <= 80)),
    sleep_hours                 REAL CHECK (sleep_hours IS NULL OR (sleep_hours >= 0 AND sleep_hours <= 24)),
    water_intake_liters         REAL CHECK (water_intake_liters IS NULL OR (water_intake_liters >= 0 AND water_intake_liters <= 20)),
    steps_count                 INTEGER CHECK (steps_count IS NULL OR (steps_count >= 0 AND steps_count <= 300000)),
    exercise_minutes            INTEGER CHECK (exercise_minutes IS NULL OR (exercise_minutes >= 0 AND exercise_minutes <= 1440)),
    calories_consumed           INTEGER CHECK (calories_consumed IS NULL OR (calories_consumed >= 0 AND calories_consumed <= 20000)),
    calories_burned             INTEGER CHECK (calories_burned IS NULL OR (calories_burned >= 0 AND calories_burned <= 20000)),
    stress_level                INTEGER CHECK (stress_level IS NULL OR (stress_level >= 1 AND stress_level <= 10)),
    mood_status                 TEXT CHECK (mood_status IS NULL OR mood_status IN ('great','good','neutral','low','bad','anxious','tired','energetic')),
    medications_taken           TEXT,
    symptoms_notes              TEXT,
    created_at                  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at                  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, measurement_date)
);
"""

SAMPLE_USERS = [
    (1, "jane@example.com", "Jane Doe"),
    (2, "bob@example.com", "Bob Smith"),
    (3, "carol@example.com", "Carol Williams"),
    (4, "dave@example.com", "Dave Brown"),
    (5, "eve@example.com", "Eve Davis"),
    (6, "frank@example.com", "Frank Miller"),
]

SAMPLE_METRICS = [
    (1, str(date.today()), 92.0, 135.0, 118, 76, 72, 36.6, 98.0, 68.5, 22.8,
     7.5, 2.0, 8500, 45, 2100, 680, 4, "good", "Vitamin D 2000 IU", "Mild headache in afternoon"),
    (1, str(date(2026, 5, 24)), 88.0, 128.0, 116, 74, 70, 36.5, 98.5, 68.3, 22.7,
     8.0, 2.2, 10200, 50, 1950, 720, 3, "great", None, None),
    (1, str(date(2026, 5, 23)), 90.0, 132.0, 115, 73, 68, 36.4, 98.3, 68.4, 22.7,
     7.8, 2.1, 9800, 48, 2050, 700, 3, "good", None, None),
    (1, str(date(2026, 5, 22)), 93.0, 138.0, 120, 78, 74, 36.7, 97.8, 68.6, 22.8,
     6.5, 1.8, 7200, 30, 2150, 620, 5, "neutral", "Vitamin D 2000 IU", None),
    (2, str(date.today()), 105.0, 160.0, 132, 85, 78, 36.8, 97.0, 82.0, 27.4,
     6.0, 1.5, 4200, 20, 2500, 400, 7, "low", "Metformin 500mg", "Fatigue, light dizziness"),
    (2, str(date(2026, 5, 24)), 110.0, 155.0, 130, 82, 76, 36.7, 97.2, 82.1, 27.4,
     5.5, 1.8, 3800, 15, 2200, 350, 8, "bad", "Metformin 500mg", "Headache, blurred vision"),
    (2, str(date(2026, 5, 23)), 108.0, 158.0, 128, 84, 75, 36.9, 97.5, 82.0, 27.4,
     6.0, 1.6, 4100, 18, 2400, 380, 7, "low", "Metformin 500mg", "Dizziness in morning"),
    (2, str(date(2026, 5, 22)), 106.0, 152.0, 126, 80, 74, 36.6, 97.8, 81.8, 27.3,
     6.5, 1.7, 4500, 22, 2300, 420, 6, "low", "Metformin 500mg", None),
    (3, str(date.today()), 85.0, 120.0, 110, 70, 65, 36.4, 99.0, 60.0, 20.1,
     8.5, 2.5, 14000, 60, 1800, 900, 2, "great", None, None),
    (3, str(date(2026, 5, 24)), 84.0, 118.0, 108, 68, 63, 36.3, 99.2, 59.8, 20.0,
     8.2, 2.4, 13500, 55, 1750, 880, 2, "great", None, None),
    (3, str(date(2026, 5, 23)), 86.0, 122.0, 112, 70, 66, 36.5, 98.8, 60.2, 20.1,
     8.0, 2.3, 12800, 50, 1900, 850, 3, "good", None, "Slight muscle soreness"),
    (4, str(date.today()), 95.0, 140.0, 125, 80, 74, 36.9, 97.8, 75.0, 24.5,
     6.5, 1.8, 6500, 30, 2300, 550, 5, "neutral", "Lisinopril 10mg", "Occasional cough"),
    (4, str(date(2026, 5, 24)), 94.0, 138.0, 122, 78, 72, 36.8, 98.0, 74.8, 24.4,
     7.0, 1.9, 6800, 35, 2250, 580, 5, "neutral", "Lisinopril 10mg", None),
    (4, str(date(2026, 5, 23)), 96.0, 142.0, 126, 82, 75, 37.0, 97.5, 75.2, 24.6,
     6.0, 1.7, 6000, 25, 2350, 520, 6, "anxious", "Lisinopril 10mg", "Mild anxiety"),
    (5, str(date.today()), 98.0, 145.0, 120, 78, 71, 37.0, 97.5, 70.2, 23.1,
     7.0, 1.9, 7200, 35, 2150, 600, 6, "anxious", "SSRI 50mg", "Anxiety in evening"),
    (5, str(date(2026, 5, 24)), 97.0, 142.0, 118, 76, 70, 36.9, 97.8, 70.0, 23.1,
     7.2, 2.0, 7500, 38, 2100, 620, 5, "neutral", "SSRI 50mg", None),
    (5, str(date(2026, 5, 23)), 99.0, 148.0, 122, 80, 73, 37.1, 97.2, 70.4, 23.2,
     6.8, 1.8, 6800, 30, 2200, 580, 6, "anxious", "SSRI 50mg", "Sleep difficulty"),
    (6, str(date.today()), 87.0, 125.0, 108, 68, 62, 36.3, 99.2, 58.0, 19.8,
     9.0, 2.8, 16000, 75, 1700, 1050, 1, "energetic", None, None),
    (6, str(date(2026, 5, 24)), 86.0, 124.0, 107, 67, 61, 36.2, 99.5, 57.8, 19.7,
     8.8, 2.7, 15500, 70, 1680, 1020, 1, "energetic", None, None),
    (6, str(date(2026, 5, 23)), 88.0, 126.0, 110, 69, 64, 36.4, 99.0, 58.2, 19.8,
     8.5, 2.6, 15000, 65, 1750, 980, 2, "great", None, None),
]


SELECT_SQL = """
    SELECT
        u.display_name,
        m.measurement_date,
        m.blood_sugar_fasting,
        m.systolic_bp,
        m.diastolic_bp,
        m.heart_rate_bpm,
        m.body_temperature_celsius,
        m.oxygen_saturation_percent,
        m.body_weight_kg,
        m.sleep_hours,
        m.steps_count,
        m.stress_level,
        m.mood_status
    FROM daily_body_metrics m
    JOIN users u ON u.id = m.user_id
    ORDER BY m.measurement_date DESC
    LIMIT ?;
"""


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    try:
        conn.executescript(SCHEMA_SQL)
        cur = conn.execute("SELECT COUNT(*) FROM users")
        if cur.fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
                SAMPLE_USERS,
            )
            conn.executemany(
                """INSERT INTO daily_body_metrics (
                    user_id, measurement_date,
                    blood_sugar_fasting, blood_sugar_post_meal,
                    systolic_bp, diastolic_bp,
                    heart_rate_bpm, body_temperature_celsius, oxygen_saturation_percent,
                    body_weight_kg, bmi,
                    sleep_hours, water_intake_liters, steps_count, exercise_minutes,
                    calories_consumed, calories_burned,
                    stress_level, mood_status,
                    medications_taken, symptoms_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                SAMPLE_METRICS,
            )
        else:
            for user in SAMPLE_USERS:
                conn.execute(
                    "INSERT OR IGNORE INTO users (id, email, display_name) VALUES (?, ?, ?)",
                    user,
                )
            cur = conn.execute("SELECT COUNT(*) FROM daily_body_metrics")
            if cur.fetchone()[0] < 20:
                conn.executemany(
                    """INSERT OR IGNORE INTO daily_body_metrics (
                        user_id, measurement_date,
                        blood_sugar_fasting, blood_sugar_post_meal,
                        systolic_bp, diastolic_bp,
                        heart_rate_bpm, body_temperature_celsius, oxygen_saturation_percent,
                        body_weight_kg, bmi,
                        sleep_hours, water_intake_liters, steps_count, exercise_minutes,
                        calories_consumed, calories_burned,
                        stress_level, mood_status,
                        medications_taken, symptoms_notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    SAMPLE_METRICS,
                )
        conn.commit()
    finally:
        conn.close()


def fetch_recent_metrics(limit=5):
    conn = get_conn()
    try:
        cur = conn.execute(SELECT_SQL, (limit,))
        return cur.fetchall()
    finally:
        conn.close()


def print_metrics(rows):
    if not rows:
        print("No records found.")
        return
    print(
        f"{'Name':<20} {'Date':<12} {'Sugar':>6} {'BP':>8} {'HR':>4} "
        f"{'Temp':>5} {'SpO2':>5} {'Weight':>7} {'Sleep':>6} {'Steps':>7} "
        f"{'Stress':>6} {'Mood':<12}"
    )
    print("-" * 110)
    for r in rows:
        bp = f"{r['systolic_bp'] or '-'}/{r['diastolic_bp'] or '-'}"
        print(
            f"{r['display_name']:<20} {str(r['measurement_date']):<12} "
            f"{r['blood_sugar_fasting'] if r['blood_sugar_fasting'] is not None else '-':>6} "
            f"{bp:>8} "
            f"{r['heart_rate_bpm'] if r['heart_rate_bpm'] is not None else '-':>4} "
            f"{r['body_temperature_celsius'] if r['body_temperature_celsius'] is not None else '-':>5} "
            f"{r['oxygen_saturation_percent'] if r['oxygen_saturation_percent'] is not None else '-':>5} "
            f"{r['body_weight_kg'] if r['body_weight_kg'] is not None else '-':>7} "
            f"{r['sleep_hours'] if r['sleep_hours'] is not None else '-':>6} "
            f"{r['steps_count'] if r['steps_count'] is not None else '-':>7} "
            f"{r['stress_level'] if r['stress_level'] is not None else '-':>6} "
            f"{r['mood_status'] if r['mood_status'] is not None else '-':<12}"
        )


if __name__ == "__main__":
    init_db()
    rows = fetch_recent_metrics(limit=50)
    print_metrics(rows)
