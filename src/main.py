import streamlit as st
import os
from PIL import Image
import langid  

SHIELD_LM_SCRIPT = r"shield_lm_script.py"
KOLORS_SCRIPT = r"kolors_script.py"
FLUX_SCRIPT = r"flux_script.py"
INTERNVL_SCRIPT = r"internvl_script.py"

st.title("LLM Prompt and Image Safety Checker")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


prompt = st.text_input("Enter your prompt:")

if prompt:
    detected_lang, _ = langid.classify(prompt)
    if detected_lang == "en":
        language = "Englenish"
    elif detected_lang == "zh":
        language = "Chinese"
    else:
        language = "Unknown"
    st.write(f"Detected language: {language}")

if st.button("Submit"):
    if prompt:
        
        st.session_state["messages"] = []

        st.session_state["messages"].append({"role": "user", "content": prompt})

        # 使用 ShieldLM 进行提示词安全检测
        st.session_state["messages"].append({"role": "assistant", "content": "Checking prompt safety with ShieldLM..."})
        shield_lm_command = f"python {SHIELD_LM_SCRIPT} '{prompt}' '{detected_lang}'"
        shield_lm_result = os.popen(shield_lm_command).read().strip()
        st.session_state["messages"].append({"role": "assistant", "content": f"Prompt Safety Check Result: {shield_lm_result}"})

        # 生成图片
        st.session_state["messages"].append({"role": "assistant", "content": "Generating image based on prompt..."})
        if detected_lang == "zh":
            kolors_command = f"python {KOLORS_SCRIPT} '{prompt}'"
            generated_image_path = os.popen(kolors_command).read().strip()
        else:
            flux_command = f"python {FLUX_SCRIPT} '{prompt}'"
            generated_image_path = os.popen(flux_command).read().strip()

        # 对生成的图片进行安全检测
        if os.path.exists(generated_image_path):
            st.session_state["messages"].append({"role": "assistant", "content": "Image generated successfully."})
            st.session_state["messages"].append({"role": "assistant", "image_path": generated_image_path}) 
            st.session_state["messages"].append({"role": "assistant", "content": "Checking image safety with InternVL..."})
            internvl_command = f"python {INTERNVL_SCRIPT} {generated_image_path}"
            internvl_result = os.popen(internvl_command).read().strip()
            st.session_state["messages"].append({"role": "assistant", "content": f"Image Safety Check Result: {internvl_result}"})
        else:
            st.session_state["messages"].append({"role": "assistant", "content": "Image generation failed."})
    else:
        st.session_state["messages"].append({"role": "assistant", "content": "Please enter a prompt."})

for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    elif message["role"] == "assistant":
        if "image_path" in message:
            st.chat_message("assistant").image(message["image_path"])
        else:
            st.chat_message("assistant").markdown(message["content"])
