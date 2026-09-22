import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
import base64
import json
from PIL import Image
import io
import requests


# -----------------------------
# CONFIG
# -----------------------------

st.set_page_config(
    page_title="AgriLens AI",
    page_icon="🌾",
    layout="wide"
)

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found. Check your .env file.")
    st.stop()

client = Groq(api_key=api_key)


# -----------------------------
# IMAGE ENCODING
# -----------------------------

def prepare_image(uploaded_file):

    image = Image.open(uploaded_file)

    image = image.convert("RGB")

    image.thumbnail((1600, 1600))

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=85
    )

    image_bytes = buffer.getvalue()

    base64_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return base64_image


# -----------------------------
# WEATHER DATA
# -----------------------------

def get_weather(location):

    geocode_url = (
        "https://geocoding-api.open-meteo.com/v1/search"
    )

    geocode_params = {
        "name": location,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    geo_response = requests.get(
        geocode_url,
        params=geocode_params,
        timeout=10
    )

    geo_data = geo_response.json()

    if "results" not in geo_data:
        return None

    place = geo_data["results"][0]

    latitude = place["latitude"]
    longitude = place["longitude"]

    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m"
        ),
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation_probability,"
            "precipitation"
        ),
        "forecast_days": 3,
        "timezone": "auto"
    }

    weather_response = requests.get(
        weather_url,
        params=weather_params,
        timeout=10
    )

    weather_data = weather_response.json()

    return {
        "place": place,
        "weather": weather_data
    }


# -----------------------------
# AI ANALYSIS
# -----------------------------

def analyze_crop(
    image_data,
    crop,
    growth_stage,
    description,
    soil_ph,
    nitrogen,
    soil_moisture,
    weather
):

    prompt = f"""
You are AgriLens AI, an agricultural crop-analysis assistant.

Analyze the crop image together with the farmer's information,
soil information, and current weather.

Crop:
{crop}

Growth stage:
{growth_stage}

Farmer description:
{description if description else "No description provided"}

Soil pH:
{soil_ph}

Nitrogen:
{nitrogen}

Soil moisture:
{soil_moisture}%

Current weather:
Temperature: {weather["temperature_2m"]} °C
Humidity: {weather["relative_humidity_2m"]} %
Rain: {weather["precipitation"]} mm
Wind speed: {weather["wind_speed_10m"]} km/h

IMPORTANT:
- Analyze only what can reasonably be inferred from the image
  and provided information.
- Do not claim a definitive disease diagnosis.
- Use "possible" or "suspected" when appropriate.
- Do not invent symptoms that are not visible or provided.
- The health score is a prototype decision-support estimate,
  not a scientific measurement.
- Recommendations must be general and safe.
- Do not recommend exact pesticide dosage or chemical treatment.
- Consider weather conditions when estimating disease, pest,
  and weather risk.
- Consider soil moisture and current weather when giving
  irrigation advice.
- Do not claim that the system physically controls irrigation.

Return ONLY valid JSON with exactly these fields:

{{
    "crop_observed": "",
    "visual_observations": [],
    "possible_issue": "",
    "crop_health_score": 0,
    "disease_risk": "Low",
    "pest_risk": "Low",
    "weather_risk": "Low",
    "early_warning": "",
    "irrigation_advice": "",
    "confidence": "Low",
    "risk_reason": "",
    "recommended_actions": [],
    "expert_confirmation_needed": true
}}

crop_health_score must be between 0 and 100.

Risk values must be:
Low, Medium, or High.

Confidence must be:
Low, Medium, or High.

Irrigation advice should be a short practical recommendation
based on soil moisture, crop stage, and weather.
"""

    response = client.chat.completions.create(

        model="qwen/qwen3.8-27b",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a cautious agricultural "
                    "decision-support AI."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/jpeg;base64,"
                                + image_data
                            )
                        }
                    }
                ]
            }
        ],

        temperature=0.2,

        response_format={
            "type": "json_object"
        }
    )

    result = response.choices[0].message.content

    return json.loads(result)


# -----------------------------
# HEADER
# -----------------------------

st.title("🌾 AgriLens AI")

st.subheader(
    "AI-Powered Crop Intelligence for Farmers"
)

st.write(
    "Scan a crop and combine visual evidence, "
    "farmer observations, soil information and "
    "weather intelligence to generate an "
    "AI-assisted crop assessment."
)

st.divider()


# -----------------------------
# CROP IMAGE
# -----------------------------

st.header("📱 Scan Your Crop")

uploaded_image = st.file_uploader(
    "📷 Take a crop photo or select one from your phone",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image:

    st.success("Crop image received!")

    st.image(
        uploaded_image,
        caption="Crop image",
        use_container_width=True
    )


# -----------------------------
# CROP INFORMATION
# -----------------------------

st.header("🌱 Crop Information")

col1, col2 = st.columns(2)

with col1:

    crop = st.selectbox(
        "Select crop",
        [
            "Rice",
            "Cotton",
            "Maize",
            "Chilli",
            "Tomato",
            "Groundnut",
            "Soybean",
            "Other"
        ]
    )

with col2:

    growth_stage = st.selectbox(
        "Growth stage",
        [
            "Seedling",
            "Vegetative",
            "Flowering",
            "Fruiting",
            "Maturity"
        ]
    )


# -----------------------------
# FARMER DESCRIPTION
# -----------------------------

st.header("🎙️ What are you noticing?")

description = st.text_area(
    "Describe what you see",
    placeholder=(
        "Example: Leaves are turning yellow "
        "and I can see small insects."
    ),
    height=120
)


# -----------------------------
# FARM LOCATION
# -----------------------------

st.header("📍 Farm Location")

farm_location = st.text_input(
    "Enter your farm location",
    placeholder="Example: Karimnagar, Telangana"
)


# -----------------------------
# SOIL INFORMATION
# -----------------------------

st.header("🌱 Soil Information")

col1, col2, col3 = st.columns(3)

with col1:

    soil_ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=7.0,
        step=0.1
    )

with col2:

    nitrogen = st.selectbox(
        "Nitrogen",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

with col3:

    soil_moisture = st.slider(
        "Soil moisture (%)",
        0,
        100,
        40
    )


# -----------------------------
# ANALYZE
# -----------------------------

st.divider()

if st.button(
    "🧠 Analyze Crop with AI",
    use_container_width=True
):

    if not farm_location:

        st.warning(
            "Please enter your farm location."
        )

        st.stop()

    if not uploaded_image:

        st.warning(
            "Please upload a crop image first."
        )

        st.stop()

    try:

        with st.spinner(
            "🧠 AgriLens is analyzing the crop..."
        ):

            # Prepare image
            image_data = prepare_image(
                uploaded_image
            )

            # Get weather
            weather_result = get_weather(
                farm_location
            )

            if not weather_result:

                st.error(
                    "Could not find that location."
                )

                st.stop()

            weather = weather_result["weather"]

            place = weather_result["place"]

            # AI analysis
            analysis = analyze_crop(
                image_data,
                crop,
                growth_stage,
                description,
                soil_ph,
                nitrogen,
                soil_moisture,
                weather["current"]
            )

        st.success(
            "AI crop analysis completed!"
        )


        # -----------------------------
        # WEATHER
        # -----------------------------

        st.header(
            "🌦️ Farm Weather Intelligence"
        )

        current = weather["current"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "🌡️ Temperature",
                f'{current["temperature_2m"]} °C'
            )

        with col2:

            st.metric(
                "💧 Humidity",
                f'{current["relative_humidity_2m"]} %'
            )

        with col3:

            st.metric(
                "🌧️ Rain",
                f'{current["precipitation"]} mm'
            )

        with col4:

            st.metric(
                "💨 Wind",
                f'{current["wind_speed_10m"]} km/h'
            )

        st.info(
            f'Weather location: {place["name"]}, '
            f'{place.get("admin1", "")}'
        )


                # -----------------------------
        # 3-DAY WEATHER FORECAST
        # -----------------------------

        st.subheader("📅 3-Day Weather Forecast")

        hourly = weather["hourly"]

        forecast_dates = []

        for i in range(len(hourly["time"])):

            date = hourly["time"][i].split("T")[0]

            if date not in forecast_dates:
                forecast_dates.append(date)

        forecast_dates = forecast_dates[:3]

        for date in forecast_dates:

            indices = [
                i
                for i, time in enumerate(hourly["time"])
                if time.startswith(date)
            ]

            if not indices:
                continue

            max_temp = max(
                hourly["temperature_2m"][i]
                for i in indices
            )

            total_rain = sum(
                hourly["precipitation"][i]
                for i in indices
            )

            max_rain_probability = max(
                hourly["precipitation_probability"][i]
                for i in indices
            )

            avg_humidity = sum(
                hourly["relative_humidity_2m"][i]
                for i in indices
            ) / len(indices)

            st.write(f"### 📅 {date}")

            fcol1, fcol2, fcol3, fcol4 = st.columns(4)

            with fcol1:
                st.metric(
                    "🌡️ Max Temperature",
                    f"{max_temp:.1f} °C"
                )

            with fcol2:
                st.metric(
                    "🌧️ Rain Probability",
                    f"{max_rain_probability}%"
                )

            with fcol3:
                st.metric(
                    "💧 Avg Humidity",
                    f"{avg_humidity:.0f}%"
                )

            with fcol4:
                st.metric(
                    "☔ Expected Rain",
                    f"{total_rain:.1f} mm"
                )

                        # -----------------------------
        # AI DECISION SUMMARY
        # -----------------------------

        st.header("🧠 AI Field Decision")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.success(
                f"🌱 Crop Health: "
                f'{analysis["crop_health_score"]}/100'
            )

            st.write(
                f"🦠 Disease Risk: "
                f'{analysis["disease_risk"]}'
            )

            st.write(
                f"🐛 Pest Risk: "
                f'{analysis["pest_risk"]}'
            )

        with summary_col2:

            st.info(
                f"🌦️ Weather Risk: "
                f'{analysis["weather_risk"]}'
            )

            st.write(
                f"💧 Irrigation: "
                f'{analysis["irrigation_advice"]}'
            )

            st.write(
                f"⚠️ Early Warning: "
                f'{analysis["early_warning"]}'
            )


        # -----------------------------
        # CROP ASSESSMENT
        # -----------------------------

        st.divider()

        st.header(
            "🌾 AgriLens Crop Assessment"
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:

            st.metric(
                "🌱 Health Score",
                f'{analysis["crop_health_score"]}/100'
            )
            st.progress(
                    analysis["crop_health_score"] / 100
                )

        with col2:

            st.metric(
                "🦠 Disease Risk",
                analysis["disease_risk"]
            )

        with col3:

            st.metric(
                "🐛 Pest Risk",
                analysis["pest_risk"]
            )

        with col4:

            st.metric(
                "🌦️ Weather Risk",
                analysis["weather_risk"]
            )

        with col5:

            st.metric(
                "🎯 Confidence",
                analysis["confidence"]
            )


        # -----------------------------
        # EARLY WARNING
        # -----------------------------

        st.divider()

        st.header(
            "⚠️ Early Warning"
        )

        st.warning(
            analysis["early_warning"]
        )

                # -----------------------------
        # SMART IRRIGATION
        # -----------------------------

        st.header(
            "💧 Smart Irrigation Advice"
        )

        st.info(
            analysis["irrigation_advice"]
        )


        # -----------------------------
        # OBSERVATIONS
        # -----------------------------

        st.subheader(
            "👁️ Visual Observations"
        )

        for observation in analysis[
            "visual_observations"
        ]:

            st.write(
                "• " + observation
            )


        # -----------------------------
        # POSSIBLE ISSUE
        # -----------------------------

        st.subheader(
            "🔍 Possible Issue"
        )

        st.warning(
            analysis["possible_issue"]
        )


        # -----------------------------
        # RISK REASON
        # -----------------------------

        st.subheader(
            "🚨 Risk Explanation"
        )

        st.write(
            analysis["risk_reason"]
        )


        # -----------------------------
        # ACTIONS
        # -----------------------------

        st.subheader(
            "💡 Recommended Next Steps"
        )

        for action in analysis[
            "recommended_actions"
        ]:

            st.write(
                "✅ " + action
            )


        # -----------------------------
        # EXPERT WARNING
        # -----------------------------

        if analysis[
            "expert_confirmation_needed"
        ]:

            st.info(
                "⚠️ This is an AI-assisted assessment. "
                "Confirm suspected disease or pest problems "
                "with a qualified agricultural expert before "
                "taking treatment decisions."
            )


    except Exception as e:

        st.error(
            "AI analysis failed."
        )

        st.code(
            str(e)
        )

        # -----------------------------
# AI FARMER CHATBOT
# -----------------------------

st.divider()

st.header("🤖 AgriLens Farmer Assistant")

st.write(
    "Ask questions about your crop, weather, soil, "
    "disease risk, or irrigation."
)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for message in st.session_state.chat_messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


user_question = st.chat_input(
    "Ask AgriLens about your crop..."
)

if user_question:

    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.write(user_question)

    chatbot_context = f"""
You are AgriLens Farmer Assistant.

Answer the farmer's question using the available
crop assessment information.

Crop:
{crop}

Growth stage:
{growth_stage}

Soil pH:
{soil_ph}

Nitrogen:
{nitrogen}

Soil moisture:
{soil_moisture}%

Farm location:
{farm_location}

Give simple, practical and safe agricultural advice.

Do not claim a definitive disease diagnosis.
Do not recommend exact pesticide dosage.
If the situation requires professional confirmation,
tell the farmer to consult an agricultural expert.

Latest AI assessment:
{analysis if "analysis" in locals() else "No crop analysis available yet."}

Farmer question:
{user_question}
"""

    try:

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                chatbot_response = client.chat.completions.create(

                    model="qwen/qwen3.8-27b",

                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful and cautious "
                                "agricultural assistant."
                            )
                        },
                        {
                            "role": "user",
                            "content": chatbot_context
                        }
                    ],

                    temperature=0.3
                )

                answer = (
                    chatbot_response
                    .choices[0]
                    .message
                    .content
                )

                st.write(answer)

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    except Exception as e:

        st.error(
            "Chatbot failed."
        )

        st.code(
            str(e)
        )