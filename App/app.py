import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='GPA Predictor — Sleep & Academic Performance',
    page_icon='🎓',
    layout='centered',
)


# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .result-good {
        background: linear-gradient(135deg, #1a9e5c, #27ae60);
        color: white; padding: 20px 28px; border-radius: 12px;
        text-align: center; margin-bottom: 16px;
    }
    .result-bad {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white; padding: 20px 28px; border-radius: 12px;
        text-align: center; margin-bottom: 16px;
    }
    .result-title { font-size: 2rem; font-weight: 700; margin: 0; }
    .result-sub   { font-size: 1rem; opacity: 0.9; margin-top: 4px; }
    .risk-good    { color: #27ae60; font-weight: 600; }
    .risk-warn    { color: #f39c12; font-weight: 600; }
    .risk-bad     { color: #e74c3c; font-weight: 600; }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #2c3e50;
        border-left: 4px solid #3498db; padding-left: 10px;
        margin: 24px 0 12px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────────

APP_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts():
    with open(os.path.join(APP_DIR, 'model', 'smote_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(APP_DIR, 'model', 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(APP_DIR, 'model', 'feature_profiles.pkl'), 'rb') as f:
        profiles = pickle.load(f)
    return model, scaler, profiles


model, scaler, profiles = load_artifacts()


# ── Constants ─────────────────────────────────────────────────────────────────

FEATURES = ['sleep_onset', 'sleep_duration', 'night_arousals',
            'screen_time', 'caffeine', 'exercise', 'stress']

FEATURE_LABELS = {
    'sleep_onset':    'Difficulty Falling Asleep',
    'sleep_duration': 'Avg Sleep Hours',
    'night_arousals': 'Waking at Night',
    'screen_time':    'Screen Time Before Bed',
    'caffeine':       'Caffeine Use',
    'exercise':       'Physical Activity',
    'stress':         'Stress Level',
}

# direction: lower_better or higher_better | (good_threshold, risk_threshold)
RISK_CONFIG = {
    'sleep_onset':    ('lower_better',  2, 3),
    'sleep_duration': ('higher_better', 3, 2),
    'night_arousals': ('lower_better',  2, 3),
    'screen_time':    ('lower_better',  2, 3),
    'caffeine':       ('lower_better',  2, 3),
    'exercise':       ('higher_better', 3, 2),
    'stress':         ('lower_better',  1, 2),
}

RISK_TIPS = {
    'sleep_onset':    ('Try winding down 30 min before bed — no screens, dim lights.',
                       'Occasional difficulty is normal.',
                       'Regular bedtime routine and limiting caffeine after 2 pm can help.'),
    'sleep_duration': ('Great sleep duration — keep it up.',
                       'Aim for at least 6-7 hours on weeknights.',
                       'Consistent 7-8 hours is linked to significantly better GPA.'),
    'night_arousals': ('Sleeping through the night — excellent.',
                       'Occasional wake-ups are fine.',
                       'Avoid heavy meals and alcohol before bed.'),
    'screen_time':    ('Minimal screen exposure before bed — great habit.',
                       'Try switching to reading instead of scrolling.',
                       'Blue light suppresses melatonin. Use Night Mode or stop 1 hour before bed.'),
    'caffeine':       ('Low caffeine intake — good for sleep quality.',
                       'Watch out for afternoon coffee affecting night sleep.',
                       'Caffeine has a 5-hour half-life — avoid after 3 pm.'),
    'exercise':       ('Active lifestyle strongly linked to better sleep and GPA.',
                       'Even 20-min walks help sleep quality.',
                       'Regular exercise is one of the strongest predictors of good GPA in this study.'),
    'stress':         ('Low stress — keep maintaining healthy boundaries.',
                       'Moderate stress — consider time-blocking study sessions.',
                       'High stress significantly impacts both sleep and performance.'),
}


# ── Input encoding maps ───────────────────────────────────────────────────────

FREQ_OPTS  = ['Never', 'Rarely (1-2 times a week)', 'Sometimes (3-4 times a week)',
              'Often (5-6 times a week)', 'Every night']
DAILY_OPTS = ['Never', 'Rarely (1-2 times a week)', 'Sometimes (3-4 times a week)',
              'Often (5-6 times a week)', 'Every day']
SLEEP_OPTS = ['Less than 4 hours', '4-5 hours', '5-6 hours',
              '6-7 hours', '7-8 hours', 'More than 8 hours']
STRESS_OPTS = ['No stress', 'Low stress', 'High stress', 'Extremely high stress']

freq_map = {
    'Never': 1, 'Rarely (1-2 times a week)': 2,
    'Sometimes (3-4 times a week)': 3, 'Often (5-6 times a week)': 4,
    'Every night': 5, 'Every day': 5,
}
sleep_map = {
    'Less than 4 hours': 1, '4-5 hours': 2, '5-6 hours': 2,
    '6-7 hours': 3, '7-8 hours': 4, 'More than 8 hours': 5,
}
stress_map = {'No stress': 1, 'Low stress': 2, 'High stress': 3, 'Extremely high stress': 4}


# ── Helper: risk level ────────────────────────────────────────────────────────

def get_risk(feature, value):
    direction, good_t, risk_t = RISK_CONFIG[feature]
    if direction == 'lower_better':
        if value <= good_t:  return 'Good',     '✅', 'risk-good'
        elif value <= risk_t: return 'Moderate', '⚠️', 'risk-warn'
        else:                 return 'Concern',  '🔴', 'risk-bad'
    else:
        if value >= good_t:  return 'Good',     '✅', 'risk-good'
        elif value >= risk_t: return 'Moderate', '⚠️', 'risk-warn'
        else:                 return 'Concern',  '🔴', 'risk-bad'


# ── Header ────────────────────────────────────────────────────────────────────

st.title('🎓 GPA Predictor')
st.markdown(
    'Understand how your **sleep habits** and **lifestyle** affect your academic performance. '
    'Enter your profile below to get a prediction and personalised insights.'
)
st.caption('Model: Random Forest + SMOTE | Trained on 996 university students | 7 behavioural features')
st.divider()


# ── Input form ────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Your Sleep & Lifestyle Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    sleep_onset    = st.selectbox('😴  Difficulty Falling Asleep',  FREQ_OPTS,   index=2)
    sleep_duration = st.selectbox('🕐  Average Sleep Hours',         SLEEP_OPTS,  index=3)
    night_arousals = st.selectbox('🌙  Waking at Night',             FREQ_OPTS,   index=2)
    screen_time    = st.selectbox('📱  Screen Time Before Bed',      FREQ_OPTS,   index=2)

with col2:
    caffeine = st.selectbox('☕  Caffeine Use',       DAILY_OPTS,  index=2)
    exercise = st.selectbox('🏃  Physical Activity',  DAILY_OPTS,  index=2)
    stress   = st.selectbox('📚  Stress Level',        STRESS_OPTS, index=1)

st.divider()

predict_btn = st.button('🔍  Predict My GPA Category', type='primary', use_container_width=True)


# ── Prediction ────────────────────────────────────────────────────────────────

if predict_btn:

    raw_values = [
        freq_map[sleep_onset],
        sleep_map[sleep_duration],
        freq_map[night_arousals],
        freq_map[screen_time],
        freq_map[caffeine],
        freq_map[exercise],
        stress_map[stress],
    ]

    X_input  = np.array([raw_values])
    X_scaled = scaler.transform(X_input)
    pred     = model.predict(X_scaled)[0]
    proba    = model.predict_proba(X_scaled)[0]
    conf     = proba[pred] * 100
    prob_good = proba[1] * 100
    prob_low  = proba[0] * 100

    # ── Result banner ─────────────────────────────────────────────────────────

    if pred == 1:
        st.markdown(
            f'<div class="result-good">'
            f'<p class="result-title">🎯 Good / Excellent GPA</p>'
            f'<p class="result-sub">Your profile aligns with high-performing students — confidence {conf:.0f}%</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-bad">'
            f'<p class="result-title">📉 Average or Below GPA</p>'
            f'<p class="result-sub">Your profile shows patterns common among lower-performing students — confidence {conf:.0f}%</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Confidence gauge + probability bars ──────────────────────────────────

    g_col, p_col = st.columns(2)

    with g_col:
        bar_color = '#27ae60' if pred == 1 else '#e74c3c'
        gauge = go.Figure(go.Indicator(
            mode='gauge+number',
            value=conf,
            number={'suffix': '%', 'font': {'size': 36, 'color': bar_color}},
            title={'text': 'Prediction Confidence', 'font': {'size': 13}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#bdc3c7'},
                'bar':  {'color': bar_color, 'thickness': 0.7},
                'bgcolor': 'white',
                'borderwidth': 1,
                'bordercolor': '#ecf0f1',
                'steps': [
                    {'range': [0,  50], 'color': '#fdf3ec'},
                    {'range': [50, 75], 'color': '#eaf4fb'},
                    {'range': [75, 100], 'color': '#eafaf1'},
                ],
            }
        ))
        gauge.update_layout(height=240, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(gauge, use_container_width=True)

    with p_col:
        prob_fig = go.Figure(go.Bar(
            x=['Good / Excellent', 'Average or Below'],
            y=[prob_good, prob_low],
            marker_color=['#27ae60', '#e74c3c'],
            text=[f'{prob_good:.1f}%', f'{prob_low:.1f}%'],
            textposition='outside',
            cliponaxis=False,
        ))
        prob_fig.update_layout(
            title='Class Probabilities',
            yaxis=dict(range=[0, 115], showgrid=True, gridcolor='#ecf0f1'),
            xaxis=dict(showgrid=False),
            showlegend=False,
            height=240,
            margin=dict(t=40, b=10, l=20, r=20),
            plot_bgcolor='white',
            paper_bgcolor='white',
        )
        st.plotly_chart(prob_fig, use_container_width=True)

    # ── Radar chart: Your profile vs GPA groups ───────────────────────────────

    st.markdown('<div class="section-header">Your Profile vs GPA Groups</div>', unsafe_allow_html=True)
    st.caption(
        'Compares your encoded values against the average profile of students with '
        'Good/Excellent GPA (green) and Average/Below GPA (red). '
        'Higher = more frequent/severe for most factors; for Sleep Hours, higher = more sleep.'
    )

    categories = [FEATURE_LABELS[f] for f in FEATURES]

    radar = go.Figure()
    for name, vals, color, fill in [
        ('Your Profile',        raw_values,                      '#3498db', 'rgba(52,152,219,0.2)'),
        ('High GPA Average',    profiles['high_gpa_mean'],       '#27ae60', 'rgba(39,174,96,0.15)'),
        ('Average/Below Avg',   profiles['low_gpa_mean'],        '#e74c3c', 'rgba(231,76,60,0.15)'),
    ]:
        vals_closed = list(vals) + [vals[0]]
        cats_closed = categories + [categories[0]]
        radar.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=cats_closed,
            fill='toself',
            fillcolor=fill,
            line=dict(color=color, width=2.5),
            name=name,
            hovertemplate='%{theta}: %{r:.2f}<extra>' + name + '</extra>',
        ))

    radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5.5],
                            tickvals=[1, 2, 3, 4, 5],
                            gridcolor='#ecf0f1', linecolor='#bdc3c7'),
            angularaxis=dict(gridcolor='#ecf0f1', linecolor='#bdc3c7'),
            bgcolor='white',
        ),
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5),
        title=dict(text='Behavioural Profile Comparison', x=0.5, font=dict(size=14)),
        height=450,
        margin=dict(t=60, b=80, l=60, r=60),
        paper_bgcolor='white',
    )
    st.plotly_chart(radar, use_container_width=True)

    # ── Risk assessment ───────────────────────────────────────────────────────

    st.markdown('<div class="section-header">Risk Assessment</div>', unsafe_allow_html=True)

    answers = [sleep_onset, sleep_duration, night_arousals,
               screen_time, caffeine, exercise, stress]

    for i, feature in enumerate(FEATURES):
        val    = raw_values[i]
        label  = FEATURE_LABELS[feature]
        answer = answers[i]
        level, icon, css_class = get_risk(feature, val)
        tip = RISK_TIPS[feature][{'Good': 0, 'Moderate': 1, 'Concern': 2}[level]]

        r1, r2, r3 = st.columns([2, 1, 4])
        with r1:
            st.markdown(f'**{label}**  \n<small>{answer}</small>', unsafe_allow_html=True)
        with r2:
            st.markdown(f'<span class="{css_class}">{icon} {level}</span>',
                        unsafe_allow_html=True)
        with r3:
            st.caption(tip)

    # ── Feature importance ────────────────────────────────────────────────────

    st.markdown('<div class="section-header">Model Feature Importance</div>', unsafe_allow_html=True)
    st.caption('How much each factor influences the model\'s GPA predictions overall (based on training data).')

    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)
    labels_sorted = [FEATURE_LABELS[FEATURES[i]] for i in sorted_idx]
    imp_sorted    = importances[sorted_idx]
    colors        = ['#e74c3c' if FEATURES[i] in ('sleep_onset', 'night_arousals', 'sleep_duration')
                     else '#3498db' for i in sorted_idx]

    imp_fig = go.Figure(go.Bar(
        x=imp_sorted,
        y=labels_sorted,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.3f}' for v in imp_sorted],
        textposition='outside',
        cliponaxis=False,
    ))
    imp_fig.update_layout(
        xaxis=dict(title='Importance Score', range=[0, max(imp_sorted) * 1.25],
                   showgrid=True, gridcolor='#ecf0f1'),
        yaxis=dict(showgrid=False),
        showlegend=False,
        height=300,
        margin=dict(t=20, b=20, l=10, r=60),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    from plotly.graph_objects import layout as go_layout
    imp_fig.add_annotation(
        x=max(imp_sorted) * 1.2, y=-0.7,
        text='🔴 Sleep   🔵 Lifestyle',
        showarrow=False, font=dict(size=11), xanchor='right',
    )
    st.plotly_chart(imp_fig, use_container_width=True)

    # ── Profile summary table ─────────────────────────────────────────────────

    with st.expander('View your encoded profile'):
        summary = pd.DataFrame({
            'Feature':           [FEATURE_LABELS[f] for f in FEATURES],
            'Your Answer':       answers,
            'Encoded (1–5)':     raw_values,
            'High GPA Avg':      [round(v, 2) for v in profiles['high_gpa_mean']],
            'Low GPA Avg':       [round(v, 2) for v in profiles['low_gpa_mean']],
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    'Impact of Insomnia on Academic Performance | '
    'Random Forest + SMOTE | base paper dataset (n=996) | '
    'Features: sleep onset, duration, night arousals, screen time, caffeine, exercise, stress'
)
