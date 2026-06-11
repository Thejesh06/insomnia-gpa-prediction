import pandas as pd


def clean_text(x):
    if isinstance(x, str):
        return x.strip().lower()
    return x


def load_and_preprocess(file_path):

    df = pd.read_csv(file_path)

    df = df.apply(lambda col: col.map(clean_text))

    # Maps column names from both datasets to unified names
    COLUMN_MAP = {
        # Base paper dataset
        "3. how often do you have difficulty falling asleep at night?": "sleep_onset",
        "4. on average, how many hours of sleep do you get on a typical day?": "sleep_duration",
        "5. how often do you wake up during the night and have trouble falling back asleep?": "night_arousals",
        "11. how often do you use electronic devices (e.g., phone, computer) before going to sleep?": "screen_time",
        "12. how often do you consume caffeine (coffee, energy drinks) to stay awake or alert?": "caffeine",
        "13. how often do you engage in physical activity or exercise?": "exercise",
        "14. how would you describe your stress levels related to academic workload?": "stress",
        "15. how would you rate your overall academic performance (gpa or grades) in the past semester?": "gpa_category",

        # Survey dataset
        "1. difficulty falling asleep": "sleep_onset",
        "2. avg sleep hours": "sleep_duration",
        "3. waking at night": "night_arousals",
        "9. screen time before bed": "screen_time",
        "10. caffeine use": "caffeine",
        "11. physical activity": "exercise",
        "12. stress level": "stress",
        "13. academic performance": "gpa_category",

        # Common demographic columns
        "department": "department",
        "gender": "gender",
        "year of study": "year",
    }

    df = df.rename(columns=lambda x: COLUMN_MAP.get(x, x))

    freq_map = {
        "never": 1,
        "rarely": 2,
        "sometimes": 3,
        "often": 4,
        "always": 5,
        "every day": 5,
        "daily": 5,
        "rarely (1-2 times a week)": 2,
        "sometimes (3-4 times a week)": 3,
        "often (5-6 times a week)": 4,
        "every night": 5,
    }

    sleep_map = {
        "less than 4 hours": 1,
        "4-5 hours": 2,
        "5-6 hours": 2,
        "6-7 hours": 3,
        "7-8 hours": 4,
        "more than 8 hours": 5,
    }

    stress_map = {
        "no stress": 1,
        "low stress": 2,
        "high stress": 3,
        "extremely high stress": 4,
    }

    for col in ["sleep_onset", "night_arousals", "screen_time", "caffeine", "exercise"]:
        if col in df.columns:
            df[col] = df[col].map(freq_map)

    if "sleep_duration" in df.columns:
        df["sleep_duration"] = df["sleep_duration"].map(sleep_map)

    if "stress" in df.columns:
        df["stress"] = df["stress"].map(stress_map)

    if "gpa_category" in df.columns:
        df["target"] = df["gpa_category"].apply(
            lambda x: 1 if x in ["good", "excellent"] else 0
        )

    return df
