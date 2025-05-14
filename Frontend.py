# 1. Environment setup (FIRST THING IN FILE)
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEB_SOCKET_COMPRESSION"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# 2. Warning filters
import warnings
warnings.filterwarnings("ignore", message="Compiled the loaded model")
warnings.filterwarnings("ignore", module="torch._classes")
warnings.filterwarnings("ignore", message="Tried to instantiate class")

# 3. Regular imports
import asyncio
import sys
if sys.platform == "win32" and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Rest of your imports
import io
from tkinter import Image, Message
from PIL import Image
import streamlit as st
import base64
from suma_Ai import AIAgent
import streamlit.components.v1 as components
from pymongo import MongoClient
import warnings

# Initialize the AI agent
ai_agent = AIAgent()
MONGO_URI = "mongodb+srv://AI_agent:z8W1L0n41kZvseDw@unisys.t75li.mongodb.net/?retryWrites=true&w=majority&appName=Unisys"


# Set Page Config
st.set_page_config(
    page_title="Vehicle Tracking System",
    layout="wide",
    page_icon="🚗",
    initial_sidebar_state="expanded"
)

def get_vehicle_count():
    try:
        client = MongoClient(MONGO_URI)
        db = client["vehicle_detection"]
        collection = db["detected_vehicles"]
        return collection.count_documents({})
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return "N/A"
    finally:
        client.close()

# def display_vehicle_info(vehicle_info):
#     cols = st.columns([1, 3])
#     with cols[0]:
#         st.markdown("🚗")
#     with cols[1]:
#         st.markdown(f"""
#         **Vehicle Information**  
#         Plate: `{vehicle_info.get('license_plate', 'Unknown')}`  
#         Type: `{vehicle_info.get('modal_type', 'Unknown').title()}`  
#         Color: `{vehicle_info.get('color', 'Unknown').title()}`  
#         Company: `{vehicle_info.get('company', 'Unknown').title()}`  
#         Starting Camera: `{vehicle_info.get('camera_number', 'Unknown')}`
#         """)
#     st.markdown("---")

# Enhanced CSS with pure white background
st.markdown("""
    <style>
        :root {
            --unisys-royal-blue: #005A9D;
            --unisys-bright-green: #00E58E;
            --unisys-dark-teal: #007173;
            --unisys-medium-blue: #007BC3;
            --text-color: #333333;
        }
        
        html, body, [class*="css"]  {
            background-color: #FFFFFF !important;
        }
        
        .main {
            background-color: #FFFFFF !important;
        }
        
        /* Sidebar Styles */
        .sidebar .sidebar-content {
            background-color: var(--unisys-royal-blue);
            color: white;
        }
        
        /* Status Cards */
        .status-card {
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        /* Chat Interface */
        .chat-container {
            height: 65vh;
            overflow-y: auto;
            background-color: #FFFFFF !important;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid #e1e4e8;
        }
        
        /* Custom upload button */
        .upload-btn {
            background-color: var(--unisys-royal-blue) !important;
            color: white !important;
            border-radius: 50% !important;
            width: 40px !important;
            height: 40px !important;
        }
        
        /* Camera feed styling */
        .camera-feed {
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 15px;
            border: 1px solid #e1e4e8;
        }
        
        /* Larger logo */
        .sidebar-logo {
            width: 180px !important;
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar with larger logo
with st.sidebar:
    st.image(r"E:\Unisys3\logo\unisyslogo.jpg", width=180)  # Increased size
    
    st.markdown("""
    <h3 style='color: white; margin-bottom: 25px;'>
        AI Vehicle Tracking System
    </h3>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: #ffffff33'>", unsafe_allow_html=True)
   
    
    # System Status
    st.markdown("<h4 style='color: white;'>System Status</h4>", unsafe_allow_html=True)
    vehicle_count = get_vehicle_count()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="status-card" style="background-color: #00E58E; color: #003D2C;">
            <b>🟢 Active Cameras</b><br>
            <h3 style="margin: 5px 0; color: #003D2C;">7/8</h3>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="status-card" style="background-color: #007173; color: white;">
            <b>🚗 Vehicles Detected</b><br>
            <h3 style="margin: 5px 0; color: white;">{vehicle_count}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: #ffffff33'>", unsafe_allow_html=True)
    
    # Quick Actions
    st.markdown("<h4 style='color: white;'>Quick Actions</h4>", unsafe_allow_html=True)
    if st.button("🔄 Refresh System", use_container_width=True, key="refresh_btn"):
        st.toast("System refreshed successfully", icon="✅")
    if st.button("⏸ Pause Monitoring", use_container_width=True, key="pause_btn"):
        st.warning("Monitoring paused - alerts disabled")

# Main Content
st.markdown("# 🚗 Live Vehicle Tracking Dashboard")

# Camera Feeds in a grid
cameras = [
    {"name": "Camera 1", "video": r"E:\Unisys3\DEMO_New\1.mp4"},
    {"name": "Camera 2", "video": r"E:\Unisys3\DEMO_New\2.mp4"},
    {"name": "Camera 3", "video": r"E:\Unisys3\DEMO_New\3.mp4"},
    {"name": "Camera 4", "video": r"E:\Unisys3\DEMO_New\4.mp4"},
    {"name": "Camera 5", "video": r"E:\Unisys3\DEMO_New\5.mp4"},
    {"name": "Camera 6", "video": r"E:\Unisys3\DEMO_New\6.mp4"},
    {"name": "Camera 8", "video": r"E:\Unisys3\DEMO_New\8.mp4"},
]

# Display cameras in a responsive grid
cols = st.columns(3)
for i, camera in enumerate(cameras):
    with cols[i % 3]:
        with st.container():
            st.markdown(f"**{camera['name']}**")
            st.video(camera['video'])
            st.markdown("<div class='camera-feed'></div>", unsafe_allow_html=True)

# --- Chat Section ---
st.markdown("---")
st.markdown("## AI Assistant")

# --- Cleanup Uploaded Image on Rerun ---
if "cleanup_required" in st.session_state and st.session_state.cleanup_required:
    if st.session_state.uploaded_image_path and os.path.exists(st.session_state.uploaded_image_path):
        os.remove(st.session_state.uploaded_image_path)
    st.session_state.uploaded_image_path = None
    st.session_state.awaiting_query = False
    st.session_state.cleanup_required = False
    st.rerun()

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help with vehicle tracking today?"}]
if "awaiting_query" not in st.session_state:
    st.session_state.awaiting_query = False
if "uploaded_image_path" not in st.session_state:
    st.session_state.uploaded_image_path = None
if "cleanup_required" not in st.session_state:
    st.session_state.cleanup_required = False
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None

# --- Chat Messages ---
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Display message text
            st.markdown(message["content"])
            
            # Display uploaded image (if available)
            if "image" in message and os.path.exists(message["image"]):
                st.image(message["image"], caption="Uploaded Image", width=300)

            # Display tracking graph (from bytes) with proper sizing
            if "graph" in message:
                try:
                    if isinstance(message["graph"], bytes):
                        img = Image.open(io.BytesIO(message["graph"]))
                        img = img.resize((600, 550))
                        st.image(img, caption="Vehicle Tracking Path")
                    else:
                        st.warning("Graph data format not recognized")
                except Exception as e:
                    st.error(f"Couldn't display tracking graph: {str(e)}")

            # # Display vehicle info if available
            # if "vehicle_info" in message:
            #     display_vehicle_info(message["vehicle_info"])

# --- Chat Input Section ---
with st.container():
    cols = st.columns([1, 8, 1])
    with cols[1]:  # Center input area
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="image_uploader")

    if st.session_state.awaiting_query:
        query = st.chat_input("Please enter your query about the image:")
        if query:
            # Append user message
            st.session_state.messages.append({
                "role": "user",
                "content": query,
                "image": st.session_state.uploaded_image_path
            })

            with chat_container:
                with st.chat_message("user"):
                    st.markdown(query)
                    st.image(st.session_state.uploaded_image_path, caption="Uploaded Image", width=500)

            # Process with AI agent
            with st.spinner("Analyzing..."):
                try:
                    response = ai_agent.process_vehicle_query(
                        "image", 
                        st.session_state.uploaded_image_path, 
                        query
                    )
                    
                    # Handle tracking initialization message
                    if isinstance(response, dict) and "tracking_init" in response:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response["tracking_init"]["content"],
                            "vehicle_info": response["tracking_init"]["vehicle_info"]
                        })
                    
                    # Handle the rest of the response
                    if isinstance(response, dict) and "graph" in response:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response["text"],
                            "graph": response["graph"]
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })

                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {str(e)}"
                    })

            # Cleanup and reset
            st.session_state.cleanup_required = True
            st.rerun()

    else:
        prompt = st.chat_input("Ask about vehicle activity...")

        # Handle either prompt or image
        if prompt or uploaded_file:
            if uploaded_file:
                # Save uploaded image and wait for query
                os.makedirs("temp_upload", exist_ok=True)
                image_path = os.path.join("temp_upload", uploaded_file.name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.session_state.awaiting_query = True
                st.session_state.uploaded_image_path = image_path
                st.rerun()

            elif prompt:
                # Append user message
                st.session_state.messages.append({"role": "user", "content": prompt})

                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)

                # Create a status container
                status_container = st.empty()
                
                with status_container:
                    with st.chat_message("assistant"):
                        st.markdown("🔍 Starting vehicle tracking...")

                # Call the AI Agent
                with st.spinner("Analyzing..."):
                    try:
                        response = ai_agent.process_vehicle_query("text", prompt, None)

                        # Clear the status container
                        status_container.empty()

                        # Handle tracking initialization
                        if isinstance(response, dict) and "tracking_init" in response:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response["tracking_init"]["content"],
                                "vehicle_info": response["tracking_init"]["vehicle_info"]
                            })
                        
                        # Handle the rest of the response
                        if isinstance(response, dict) and "graph" in response:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response["text"],
                                "graph": response["graph"]
                            })
                        else:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response
                            })
                        
                        st.rerun()
                        
                    except Exception as e:
                        status_container.empty()
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error: {str(e)}"
                        })
                        st.rerun()

                        
                # Append assistant message
                st.session_state.messages.append({"role": "assistant", "content": response})

                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown(response)