import streamlit as st
import streamlit.components.v1 as components
import paramiko
import os
import sys
import cv2
import numpy as np
import subprocess
import time
from collections import deque
from datetime import datetime
import shlex
import psutil
import pywhatkit as kit
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFilter
import io
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# --- Dependency Check for Optional Libraries ---
# (This section remains the same)

# --- Page Configuration ---
st.set_page_config(page_title="Dev Control Center", page_icon="⚙️", layout="wide")


# --- Application Mode 1: Remote Docker Manager ---
def run_remote_manager():
    """Contains all logic and UI for the SSH-based remote manager."""

    st.title("🐳 Remote Docker Manager")
    st.markdown("Manage your remote Docker environment with ease via SSH.")

    # --- SSH Connection Setup in Sidebar ---
    st.sidebar.markdown("### 🔐 SSH Connection")
    host = st.sidebar.text_input("SSH Host", placeholder="e.g., 192.168.1.10", key="ssh_host")
    username = st.sidebar.text_input("SSH Username", placeholder="e.g., admin", key="ssh_user")
    password = st.sidebar.text_input("SSH Password", type="password", key="ssh_pass")

    # --- Cached Function to Manage SSH Connection ---
    @st.cache_resource(ttl=3600) # Cache the connection for 1 hour
    def get_ssh_client(_host, _user, _pass):
        """Creates and caches a Paramiko SSH client."""
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(hostname=_host, username=_user, password=_pass, timeout=5)
            return ssh
        except Exception as e:
            # Reraise the exception to be caught later
            raise e

    def run_remote_command(ssh_client, cmd):
        """Executes a command on the remote host using the provided client."""
        try:
            stdin, stdout, stderr = ssh_client.exec_command(cmd)
            out = stdout.read().decode().strip()
            err = stderr.read().decode().strip()
            exit_code = stdout.channel.recv_exit_status()
            return out, err, exit_code
        except Exception as e:
            return None, f"Command Execution ERROR: {e}", -1

    # --- Main Logic ---
    if not (host and username and password):
        st.info("ℹ️ Please enter your SSH credentials in the sidebar to connect.")
        return

    # --- Establish Connection ---
    try:
        with st.spinner("Connecting to host..."):
            ssh_client = get_ssh_client(host, username, password)
            # Test connection with a simple command
            _, err, code = run_remote_command(ssh_client, "echo Connection test successful")
            if code != 0:
                raise Exception(f"Connection test failed: {err}")
        st.success("✅ SSH Connection Successful!")
        if st.sidebar.button("Disconnect"):
            st.cache_resource.clear()
            st.rerun()

    except Exception as e:
        st.error(f"🔴 Connection Failed: {e}")
        st.cache_resource.clear() # Clear cache on failure
        return

    # --- Helper functions to get docker items ---
    def get_docker_items(command):
        output, error, code = run_remote_command(ssh_client, command)
        if code == 0 and output:
            return output.splitlines()
        return []

    # --- UI for Docker Actions ---
    option = st.selectbox("Choose a Docker action:", [
        "List Containers", "🖼️ List Images", "Run Container", "▶️ Start Container", 
        "⏹️ Stop Container", "🗑️ Remove Container", "📥 Pull Image", "🗑️ Remove Image", 
        "🧹 System Cleanup"
    ])
    st.markdown("---")

    # --- Action Implementations ---

    if option == "List Containers":
        if st.button("Show All Containers"):
            with st.spinner("Loading containers..."):
                cmd = "docker ps -a --format 'table {{.Names}}\\t{{.Image}}\\t{{.Status}}\\t{{.Ports}}'"
                output, error, code = run_remote_command(ssh_client, cmd)
                if code == 0 and output:
                    st.code(output, language='bash')
                elif error:
                    st.error(f"Error: {error}")
                else:
                    st.warning("⚠️ No containers found.")

    elif option == "🖼️ List Images":
        if st.button("Show All Images"):
            with st.spinner("Loading images..."):
                cmd = "docker images --format 'table {{.Repository}}\\t{{.Tag}}\\t{{.ID}}\\t{{.Size}}'"
                output, error, code = run_remote_command(ssh_client, cmd)
                if code == 0 and output:
                    st.code(output, language='bash')
                elif error:
                    st.error(f"Error: {error}")
                else:
                    st.warning("⚠️ No images found.")

    elif option == "Run Container":
        with st.form("run_container_form"):
            st.subheader("Run a New Container")
            image_name = st.text_input("Image Name & Tag*", placeholder="e.g., nginx:latest")
            container_name = st.text_input("Container Name (optional)", placeholder="e.g., my-web-server")
            ports = st.text_input("Port Mapping (optional)", placeholder="e.g., 8080:80")
            
            submitted = st.form_submit_button("Run Container")
            if submitted and image_name:
                with st.spinner(f"Running {image_name}..."):
                    cmd = f"docker run -d"
                    if container_name:
                        cmd += f" --name {shlex.quote(container_name)}"
                    if ports:
                        cmd += f" -p {shlex.quote(ports)}"
                    cmd += f" {shlex.quote(image_name)}"
                    
                    output, error, code = run_remote_command(ssh_client, cmd)
                    if code == 0:
                        st.success("Container started successfully!")
                        st.code(output, language='bash')
                    else:
                        st.error(f"Failed to run container: {error}")

    elif option in ["▶️ Start Container", "⏹️ Stop Container", "🗑️ Remove Container"]:
        st.subheader(option)
        
        # Determine which containers to list based on the action
        if option == "▶️ Start Container":
            containers = get_docker_items("docker ps -a -f status=exited --format '{{.Names}}'")
            action_word = "start"
            if not containers:
                st.info("No stopped containers to start.")
                return
        elif option == "⏹️ Stop Container":
            containers = get_docker_items("docker ps -f status=running --format '{{.Names}}'")
            action_word = "stop"
            if not containers:
                st.info("No running containers to stop.")
                return
        else: # Remove Container
            containers = get_docker_items("docker ps -a --format '{{.Names}}'")
            action_word = "remove"
            if not containers:
                st.info("No containers to remove.")
                return

        container_to_act = st.selectbox(f"Select a container to {action_word}:", containers)
        
        if st.button(f"{option.split(' ')[0]} {action_word.capitalize()} Selected Container"):
            if container_to_act:
                with st.spinner(f"{action_word.capitalize()}ing {container_to_act}..."):
                    cmd = f"docker {action_word} {shlex.quote(container_to_act)}"
                    output, error, code = run_remote_command(ssh_client, cmd)
                    if code == 0:
                        st.success(f"Container '{output}' {action_word}ed successfully.")
                    else:
                        st.error(f"Failed to {action_word} container: {error}")

    elif option == "📥 Pull Image":
        with st.form("pull_image_form"):
            st.subheader("Pull an Image from a Registry")
            image_to_pull = st.text_input("Image Name & Tag", placeholder="e.g., ubuntu:22.04")
            submitted = st.form_submit_button("Pull Image")
            if submitted and image_to_pull:
                with st.spinner(f"Pulling {image_to_pull}..."):
                    cmd = f"docker pull {shlex.quote(image_to_pull)}"
                    output, error, code = run_remote_command(ssh_client, cmd)
                    if code == 0:
                        st.success(f"Image '{image_to_pull}' pulled successfully.")
                        st.code(output)
                    else:
                        st.error(f"Failed to pull image: {error}")

    elif option == "🗑️ Remove Image":
        st.subheader("Remove a Docker Image")
        images = get_docker_items("docker images --format '{{.Repository}}:{{.Tag}}'")
        if not images:
            st.info("No images to remove.")
            return
        
        image_to_remove = st.selectbox("Select an image to remove:", images)
        if st.button("Remove Selected Image", type="primary"):
            if image_to_remove:
                with st.spinner(f"Removing {image_to_remove}..."):
                    cmd = f"docker rmi {shlex.quote(image_to_remove)}"
                    output, error, code = run_remote_command(ssh_client, cmd)
                    if code == 0:
                        st.success(f"Image '{image_to_remove}' removed successfully.")
                        st.code(output)
                    else:
                        st.error(f"Failed to remove image: {error}")

    elif option == "🧹 System Cleanup":
        st.subheader("Cleanup Docker System")
        st.warning("⚠️ This will remove all stopped containers, all networks not used by at least one container, all dangling images, and all build cache. This action is irreversible.")
        if st.button("Run System Prune", type="primary"):
            with st.spinner("Pruning Docker system..."):
                cmd = "docker system prune -af"
                output, error, code = run_remote_command(ssh_client, cmd)
                if code == 0:
                    st.success("Docker system pruned successfully!")
                    st.code(output)
                else:
                    st.error(f"Failed to prune system: {error}")


# --- Application Mode 2: Gesture Docker Controller ---
def run_gesture_controller():
    """Contains all logic and UI for the local gesture controller."""

    # Check for mediapipe availability
    try:
        import mediapipe
        mediapipe_available = True
    except ImportError:
        mediapipe_available = False
    
    st.title("✋ Gesture Docker Controller")
    st.markdown("Control a **local** Docker container using hand gestures.")

    if not mediapipe_available:
        st.error("⚠️ MediaPipe library is required for gesture control but is not installed or failed to import.")
        st.code("pip install mediapipe")
        return
    st.info("Gesture controller UI and logic would be displayed here.")
    # The full implementation for gesture control would go here.


# --- Application Mode 3: Linux Terminal Simulator ---
def run_linux_simulator():
    """Contains all logic and UI for the Linux Terminal Simulator."""
    
    st.title("👨‍💻 Interactive Linux Terminal Simulator")
    st.write("Practice common Linux commands in a safe, simulated environment.")
    
    # Initialize session state for this mode
    if 'term_command_history' not in st.session_state: st.session_state.term_command_history = []
    if 'term_current_directory' not in st.session_state: st.session_state.term_current_directory = '/home/user'
    if 'term_directory_structure' not in st.session_state:
        st.session_state.term_directory_structure = {
            '/home/user': ['Documents', 'Downloads', 'Pictures', 'Desktop', 'README.txt'],
            '/home/user/Documents': ['report.docx', 'notes.txt', 'projects'],
            '/home/user/Documents/projects': ['alpha', 'beta'],
            '/home/user/Downloads': ['installer.dmg', 'archive.zip'],
            '/home/user/Pictures': ['photo.jpg'],
            '/': ['home', 'var', 'etc', 'bin'],
        }

    def get_items_in_path(path):
        return st.session_state.term_directory_structure.get(path, [])
    
    def get_dirs_in_path(path):
        # A simple way to differentiate: items without '.' are considered directories
        return [item for item in get_items_in_path(path) if '.' not in item]

    def execute_safe_command(command):
        """More robust command simulation."""
        cmd_parts = shlex.split(command.strip())
        if not cmd_parts: return ""
        base_cmd = cmd_parts[0]
        current_dir = st.session_state.term_current_directory

        # Simulated commands
        if base_cmd == 'pwd': return current_dir
        elif base_cmd == 'whoami': return "user"
        elif base_cmd == 'date': return datetime.now().strftime("%a %b %d %H:%M:%S IST %Y")
        elif base_cmd == 'clear':
            st.session_state.term_command_history = []
            return "Terminal cleared."
        elif base_cmd == 'echo': return " ".join(cmd_parts[1:])

        elif base_cmd == 'ls':
            items = get_items_in_path(current_dir)
            dirs = get_dirs_in_path(current_dir)
            if '-la' in cmd_parts or '-l' in cmd_parts:
                output = [f"drwxr-xr-x user user 4096 Aug 13 14:00 {d}/" for d in dirs]
                output.extend([f"-rw-r--r-- user user 1024 Aug 13 14:00 {f}" for f in items if f not in dirs])
                return "\n".join(output) if output else ""
            else:
                formatted = [f"{item}/" if item in dirs else item for item in items]
                return "  ".join(formatted) if formatted else ""

        elif base_cmd == 'cd':
            if len(cmd_parts) < 2: return "Usage: cd <directory>"
            target = cmd_parts[1]
            if target == '..': new_path = os.path.dirname(current_dir) if current_dir != '/' else '/'
            elif target == '~': new_path = '/home/user'
            elif target == '/': new_path = '/'
            else:
                if target in get_dirs_in_path(current_dir):
                    new_path = os.path.join(current_dir, target)
                else:
                    return f"bash: cd: {target}: No such directory"
            st.session_state.term_current_directory = new_path
            return ""

        elif base_cmd == 'cat':
            if len(cmd_parts) < 2: return "Usage: cat <file>"
            target_file = cmd_parts[1]
            if target_file in get_items_in_path(current_dir) and target_file not in get_dirs_in_path(current_dir):
                return f"Simulated content of {target_file}.\nThis is a text file.\nHave a nice day."
            else:
                return f"cat: {target_file}: No such file or directory"

        else:
            return f"Command not found: {base_cmd}. Try 'ls', 'cd', 'pwd', etc."

    # --- UI Layout for the Terminal ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📍 {st.session_state.term_current_directory}")
        # Display history as a scrollable area
        with st.container(height=400, border=True):
            for entry in st.session_state.term_command_history:
                st.code(entry['cmd'], language='bash')
                if entry['out']: # Only show output area if there is output
                    st.text(entry['out'])

        # Command input using a form
        with st.form(key='terminal_form', clear_on_submit=True):
            command = st.text_input("Enter command:", placeholder="Type a command and press Execute...", label_visibility="collapsed")
            submitted = st.form_submit_button("Execute")

        if submitted and command:
            output = execute_safe_command(command)
            st.session_state.term_command_history.append({'cmd': f"$ {command}", 'out': output})
            st.rerun()

    with col2:
        st.subheader("⚡ Quick Commands")
        st.write("Click a command to run it instantly.")
        
        # --- List of all clickable commands ---
        quick_commands = [
            "ls", "ls -la", "pwd", "whoami", "date", 
            "cd ..", "cd ~", "cd Documents",
            "cat README.txt", "echo 'Hello World!'"
        ]
        
        for cmd in quick_commands:
            if st.button(cmd, use_container_width=True):
                output = execute_safe_command(cmd)
                st.session_state.term_command_history.append({'cmd': f"$ {cmd}", 'out': output})
                st.rerun()

    # --- UI Layout for the Terminal ---
    st.info(f"Current Directory: {st.session_state.term_current_directory}")

    # Display history
    with st.container(height=350, border=True):
        for entry in st.session_state.term_command_history:
            st.code(entry['cmd'], language='bash')
            if entry['out']: # Only display output if there is any
                st.text(entry['out'])
    
    # Command input
    command = st.chat_input("Type a command and press Enter...")
    if command:
        output = execute_safe_command(command)
        st.session_state.term_command_history.append({'cmd': f"$ {command}", 'out': output})
        st.rerun()


# --- App Mode 4: Python Power Tools ---
def run_python_menu():
    """A suite of powerful tools built with Python libraries."""
    
    st.title("🐍 Python Power Tools")
    st.markdown("A collection of useful utilities. Choose a tool from the sidebar.")

    # --- Custom CSS for this page ---
    st.markdown("""
    <style>
    .tool-card {
        border: 1px solid #2e3b4e;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #1a202c;
    }
    .tool-header {
        font-size: 1.5em;
        font-weight: bold;
        color: #38b2ac;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Sidebar Navigation for Python Tools ---
    with st.sidebar:
        st.header("Tools Menu")
        page = st.radio(
            "Select a tool:",
            [
                "💻 System Monitor", "📱 WhatsApp Sender", "📞 Call & SMS", "📧 Email Sender", 
                "📸 Instagram Poster", "🌐 Web Utilities", "🎨 Image Studio", "🔄 Face Swap"
            ],
            label_visibility="collapsed"
        )

    # --- Page Implementations ---
    if page == "💻 System Monitor":
        st.markdown('<div class="tool-card"><p class="tool-header">System Resource Monitor</p></div>', unsafe_allow_html=True)
        if st.button("📊 Show System Stats"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("CPU")
                cpu_percent = psutil.cpu_percent(interval=1)
                st.metric("Usage", f"{cpu_percent}%")
                st.progress(cpu_percent / 100)
            with col2:
                st.subheader("Memory")
                mem = psutil.virtual_memory()
                st.metric("Usage", f"{mem.percent}%")
                st.progress(mem.percent / 100)
            with col3:
                st.subheader("Disk")
                disk = psutil.disk_usage('/')
                st.metric("Usage", f"{disk.percent}%")
                st.progress(disk.percent / 100)

    elif page == "📱 WhatsApp Sender":
        st.markdown('<div class="tool-card"><p class="tool-header">WhatsApp Message Sender</p></div>', unsafe_allow_html=True)
        st.info("This tool automates WhatsApp Web. You must be logged in on your default browser.")
        with st.form("whatsapp_form"):
            phone_number = st.text_input("📞 Phone Number (with country code)", "+91")
            message = st.text_area("💬 Message")
            if st.form_submit_button("🚀 Send Message"):
                if phone_number and message:
                    try:
                        with st.spinner("Opening WhatsApp Web..."):
                            kit.sendwhatmsg_instantly(phone_number, message)
                        st.success("✅ Message sent successfully!")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                else:
                    st.warning("Please provide a phone number and a message.")

    elif page == "📞 Call & SMS":
        st.markdown('<div class="tool-card"><p class="tool-header">Twilio Call & SMS Service</p></div>', unsafe_allow_html=True)
        # Check for Twilio availability and import Client
        try:
            from twilio.rest import Client
            twilio_available = True
        except ImportError:
            twilio_available = False

        if not twilio_available:
            st.error("Twilio library not found. Please run `pip install twilio`.")
            return

        with st.expander("🔐 Enter Twilio Credentials", expanded=True):
            account_sid = st.text_input("Twilio Account SID", type="password")
            auth_token = st.text_input("Twilio Auth Token", type="password")
            twilio_number = st.text_input("Your Twilio Phone Number")

        tab1, tab2 = st.tabs(["📞 Make a Call", "✉️ Send SMS"])
        with tab1:
            with st.form("call_form"):
                to_number_call = st.text_input("Recipient's Number (for Call)")
                message_to_say = st.text_area("Message to Speak", "Hello from Streamlit!")
                if st.form_submit_button("🚀 Place Call"):
                    try:
                        client = Client(account_sid, auth_token)
                        twiml = f'<Response><Say>{message_to_say}</Say></Response>'
                        call = client.calls.create(to=to_number_call, from_=twilio_number, twiml=twiml)
                        st.success(f"✅ Call initiated! SID: {call.sid}")
                    except Exception as e:
                        st.error(f"❌ Call Failed: {e}")
        with tab2:
            with st.form("sms_form"):
                to_number_sms = st.text_input("Recipient's Number (for SMS)")
                message_body = st.text_area("SMS Message")
                if st.form_submit_button("🚀 Send SMS"):
                    try:
                        client = Client(account_sid, auth_token)
                        message = client.messages.create(body=message_body, from_=twilio_number, to=to_number_sms)
                        st.success(f"✅ SMS sent! SID: {message.sid}")
                    except Exception as e:
                        st.error(f"❌ SMS Failed: {e}")

    elif page == "📧 Email Sender":
        st.markdown('<div class="tool-card"><p class="tool-header">Gmail Email Sender</p></div>', unsafe_allow_html=True)
        st.warning("💡 Use a **Gmail App Password** for this to work, not your regular password.")
        with st.form("gmail_form"):
            sender_email = st.text_input("📤 Your Gmail Address")
            app_password = st.text_input("🔑 Your Gmail App Password", type="password")
            recipient_email = st.text_input("📥 Recipient's Email")
            subject = st.text_input("📝 Subject")
            body = st.text_area("💬 Email Body")
            if st.form_submit_button("📧 Send Email"):
                try:
                    msg = MIMEMultipart()
                    msg["From"] = sender_email
                    msg["To"] = recipient_email
                    msg["Subject"] = subject
                    msg.attach(MIMEText(body, "plain"))
                    with smtplib.SMTP("smtp.gmail.com", 587) as server:
                        server.starttls()
                        server.login(sender_email, app_password)
                        server.send_message(msg)
                    st.success("✅ Email sent successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to send email: {e}")

    # ... Other pages like Instagram, Web Utilities, Image Studio, Face Swap would follow a similar redesigned structure ...

# --- App Mode 5: Kubernetes Dashboard ---
def run_kubernetes_dashboard():
    """A simple dashboard to interact with a local Kubernetes cluster."""
    
    st.title("🚢 Kubernetes Dashboard")
    st.markdown("Interact with your local Kubernetes cluster (Minikube, Docker Desktop, etc.)")

    # Check for kubernetes library and import if available
    try:
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
        kubernetes_available = True
    except ImportError:
        kubernetes_available = False

    if not kubernetes_available:
        st.error("The `kubernetes` Python library is not installed. Please run:")
        st.code("pip install kubernetes", language="bash")
        return

    # --- Load Kubeconfig and Initialize API Clients ---
    try:
        config.load_kube_config()
        v1 = client.CoreV1Api()
        apps_v1 = client.AppsV1Api()
        st.success("✅ Successfully connected to Kubernetes cluster.")
    except Exception as e:
        st.error(f"❌ Could not load Kubernetes configuration: {e}")
        st.warning("Please ensure your `kubeconfig` file is correctly set up.")
        return

    st.markdown("---")
    
    # Use a selectbox for resource navigation instead of a conflicting sidebar
    menu_option = st.selectbox(
        "Select a resource to manage:",
        ("List Pods", "Create Pod", "Delete Pod", "List Deployments", "List Services", "List Nodes")
    )

    if menu_option == "List Pods":
        st.subheader("📜 Pods in 'default' namespace")
        try:
            pods = v1.list_namespaced_pod(namespace="default")
            pod_data = [{"Name": p.metadata.name, "Status": p.status.phase, "Node": p.spec.node_name, "IP": p.status.pod_ip} for p in pods.items]
            st.dataframe(pod_data, use_container_width=True)
        except ApiException as e:
            st.error(f"Error listing pods: {e}")

    elif menu_option == "Create Pod":
        st.subheader("➕ Create a New Pod")
        with st.form("create_pod_form"):
            pod_name = st.text_input("Pod Name", "my-new-pod")
            pod_image = st.text_input("Container Image", "nginx:latest")
            if st.form_submit_button("Create Pod"):
                pod_manifest = {
                    "apiVersion": "v1", "kind": "Pod",
                    "metadata": {"name": pod_name},
                    "spec": {"containers": [{"name": pod_name, "image": pod_image}]}
                }
                try:
                    v1.create_namespaced_pod(namespace="default", body=pod_manifest)
                    st.success(f"✅ Pod '{pod_name}' created successfully!")
                except ApiException as e:
                    st.error(f"Error creating pod: {e.body}")

    elif menu_option == "Delete Pod":
        st.subheader("🗑️ Delete a Pod")
        try:
            pods = v1.list_namespaced_pod(namespace="default").items
            pod_list = [p.metadata.name for p in pods]
            if not pod_list:
                st.warning("No pods found in the 'default' namespace to delete.")
            else:
                pod_to_delete = st.selectbox("Select Pod to delete", pod_list)
                if st.button("Delete Pod", type="primary"):
                    v1.delete_namespaced_pod(name=pod_to_delete, namespace="default")
                    st.success(f"🗑️ Pod '{pod_to_delete}' deleted successfully!")
                    st.rerun()
        except ApiException as e:
            st.error(f"Error deleting pod: {e.body}")

    elif menu_option == "List Deployments":
        st.subheader("📦 Deployments in 'default' namespace")
        try:
            deployments = apps_v1.list_namespaced_deployment(namespace="default")
            dep_data = [{"Name": d.metadata.name, "Replicas": d.spec.replicas, "Available": d.status.available_replicas or 0} for d in deployments.items]
            st.dataframe(dep_data, use_container_width=True)
        except ApiException as e:
            st.error(f"Error listing deployments: {e}")

    elif menu_option == "List Services":
        st.subheader("🔌 Services in 'default' namespace")
        try:
            services = v1.list_namespaced_service(namespace="default")
            svc_data = [{"Name": s.metadata.name, "Type": s.spec.type, "Cluster IP": s.spec.cluster_ip, "Ports": str(s.spec.ports)} for s in services.items]
            st.dataframe(svc_data, use_container_width=True)
        except ApiException as e:
            st.error(f"Error listing services: {e}")

    elif menu_option == "List Nodes":
        st.subheader("💻 Cluster Nodes")
        try:
            nodes = v1.list_node()
            node_data = [{"Name": n.metadata.name, "Status": n.status.conditions[-1].type, "Kubelet Version": n.status.node_info.kubelet_version} for n in nodes.items]
            st.dataframe(node_data, use_container_width=True)
        except ApiException as e:
            st.error(f"Error listing nodes: {e}")


# --- App Mode 6: Git Automation ---
def run_git_automation():
    """A UI to automate common Git and GitHub workflows."""
    st.title("🐙 Git & GitHub Automation")
    st.markdown("Automate repository creation, cloning, commits, and pushes.")
    st.warning("This tool requires `git` to be installed on the machine running this app.")

    # --- Helper Functions ---
    def run_command(command, cwd, log_area):
        log_area.info(f"▶️ Running: {' '.join(command)}")
        try:
            result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, encoding='utf-8')
            if result.stdout: log_area.code(result.stdout, language='bash')
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            log_area.error(f"❌ Command failed:\n{e.stderr}")
            return False, e.stderr

    def api_request(method, url, headers, json=None):
        try:
            response = requests.request(method, url, headers=headers, json=json)
            response.raise_for_status()
            return (True, response.json()) if response.content else (True, {})
        except requests.exceptions.HTTPError as e:
            return False, e.response.json().get('message', e.response.text)

    # --- Session State ---
    if 'git_workspace' not in st.session_state:
        st.session_state.git_workspace = tempfile.mkdtemp(prefix="git_automation_")
        st.session_state.local_repos = {} # name: path

    workspace = st.session_state.git_workspace

    # --- Main UI ---
    with st.expander("⚙️ Configuration & Credentials", expanded=True):
        github_username = st.text_input("GitHub Username", key="github_username")
        github_token = st.text_input("GitHub Personal Access Token", type="password", key="github_token", help="Requires `repo`, `workflow`, and `delete_repo` scopes.")
        auth_ready = github_username and github_token
        if auth_ready: st.success("Credentials Ready!")

    headers = {"Authorization": f"token {github_token}", "Accept": "application/vnd.github.v3+json"}
    
    tab1, tab2, tab3 = st.tabs(["🚀 Create & Push New Repo", "✏️ Clone & Commit Changes", "🗑️ Delete Repo"])

    with tab1:
        st.subheader("1. Create a New Repository on GitHub")
        with st.form("create_repo_form"):
            repo_name = st.text_input("New Repository Name")
            repo_desc = st.text_area("Description (optional)")
            is_private = st.checkbox("Make repository private")
            create_repo_btn = st.form_submit_button("Create on GitHub", disabled=not auth_ready)
        
        if create_repo_btn and repo_name:
            with st.spinner(f"Creating '{repo_name}' on GitHub..."):
                payload = {"name": repo_name, "description": repo_desc, "private": is_private}
                success, resp = api_request("POST", "https://api.github.com/user/repos", headers, json=payload)
                if success:
                    st.success(f"✅ Repository '{repo_name}' created on GitHub!")
                    clone_url = resp.get('clone_url')
                    st.code(f"git clone {clone_url}", language="bash")
                else:
                    st.error(f"❌ Failed to create repo: {resp}")

    with tab2:
        st.subheader("2. Clone, Create File, Commit, and Push")
        
        # Clone
        clone_url = st.text_input("GitHub Repo URL to Clone (e.g., https://github.com/user/repo.git)")
        if st.button("Clone Repository", disabled=not auth_ready):
            if clone_url:
                repo_name = clone_url.split('/')[-1].replace('.git', '')
                local_path = str(Path(workspace) / repo_name)
                log_area = st.container(border=True)
                # Use token in URL for private repos
                authed_url = clone_url.replace("https://", f"https://{github_username}:{github_token}@")
                success, _ = run_command(["git", "clone", authed_url, local_path], workspace, log_area)
                if success:
                    st.session_state.local_repos[repo_name] = local_path
                    st.success(f"Cloned '{repo_name}' to workspace.")
        
        # Commit and Push
        if st.session_state.local_repos:
            repo_to_modify = st.selectbox("Select a cloned repo to modify:", list(st.session_state.local_repos.keys()))
            with st.form("commit_form"):
                file_name = st.text_input("File to create/overwrite", "hello.txt")
                file_content = st.text_area("File content", f"Hello from Streamlit at {datetime.now().isoformat()}")
                commit_message = st.text_input("Commit Message", "Automated commit from Streamlit")
                commit_push_btn = st.form_submit_button("Commit and Push", disabled=not auth_ready)

            if commit_push_btn and repo_to_modify:
                log_area = st.container(border=True)
                repo_path = st.session_state.local_repos[repo_to_modify]
                # Create the file
                with open(Path(repo_path) / file_name, "w") as f:
                    f.write(file_content)
                log_area.info(f"Created file '{file_name}'.")

                # Run Git commands
                run_command(["git", "config", "user.name", github_username], repo_path, log_area)
                run_command(["git", "config", "user.email", f"{github_username}@users.noreply.github.com"], repo_path, log_area)
                run_command(["git", "add", file_name], repo_path, log_area)
                run_command(["git", "commit", "-m", commit_message], repo_path, log_area)
                success, _ = run_command(["git", "push"], repo_path, log_area)
                if success: st.success("✅ Changes pushed successfully!")

    with tab3:
        st.subheader("3. Delete a Repository from GitHub")
        st.warning("🔴 This action is irreversible and permanently deletes the repository from GitHub.", icon="⚠️")
        repo_to_delete = st.text_input("Enter the exact name of the repository to delete")
        if st.checkbox(f"I understand I am about to permanently delete '{repo_to_delete}'."):
            if st.button("Delete from GitHub", disabled=not auth_ready, type="primary"):
                if repo_to_delete:
                    with st.spinner(f"Deleting '{repo_to_delete}'..."):
                        url = f"https://api.github.com/repos/{github_username}/{repo_to_delete}"
                        success, resp = api_request("DELETE", url, headers)
                        if success:
                            st.success(f"✅ Repository '{repo_to_delete}' deleted from GitHub.")
                            if repo_to_delete in st.session_state.local_repos:
                                shutil.rmtree(st.session_state.local_repos[repo_to_delete])
                                del st.session_state.local_repos[repo_to_delete]
                        else:
                            st.error(f"❌ Failed to delete repo: {resp}")
                else:
                    st.warning("Please enter a repository name.")

# --- App Mode 7: Linear Regression ---
def run_linear_regression():
    """A simple ML model to predict temperature."""
    st.title("📈 Linear Regression: Temperature Prediction")
    st.markdown("Use a simple machine learning model to predict temperature based on weather inputs.")

    # --- Create sample data in memory (no external file needed) ---
    csv_data = """Humidity,Wind_Speed,Previous_Temp,Today_Temp
60,10,22,24
65,12,24,25
70,18,25,26
55,5,23,24
75,15,26,27
80,32,27,28
62,19,24,25
58,18,22,23
72,26,25,26
68,9,24,25
"""
    df = pd.read_csv(io.StringIO(csv_data))

    # --- Sidebar for user inputs ---
    st.sidebar.header("Enter Weather Details")
    humidity = st.sidebar.slider("Humidity (%)", 50, 85, 65)
    wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 20, 10)
    previous_temp = st.sidebar.number_input("Previous Day Temp (°C)", min_value=15.0, max_value=35.0, value=24.0)

    # --- Model Training and Prediction ---
    x = df[["Humidity", "Wind_Speed", "Previous_Temp"]]
    y = df["Today_Temp"]
    
    model = LinearRegression()
    model.fit(x, y)
    
    predicted_temp = model.predict([[humidity, wind_speed, previous_temp]])

    # --- Display Prediction ---
    st.subheader("Prediction Result")
    st.metric(label="Predicted Temperature", value=f"{predicted_temp[0]:.2f} °C")

    with st.expander("Show Model Details"):
        st.write("This is a simple Linear Regression model trained on sample data.")
        st.write(f"**Model Intercept:** `{model.intercept_:.2f}`")
        st.write(f"**Model Coefficients:** `{model.coef_}`")

    # --- Graphing and Data View ---
    st.markdown("---")
    st.subheader("📊 Historical Data Visualization")
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(df["Humidity"], df["Wind_Speed"], c=df["Today_Temp"], cmap='viridis', s=100, alpha=0.7)
    ax.set_xlabel("Humidity (%)")
    ax.set_ylabel("Wind Speed (km/h)")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Today's Temp (°C)")
    st.pyplot(fig)

    with st.expander("View Raw Data"):
        st.dataframe(df)


# --- App Mode 8: Web Playground ---
def run_javascript_menu():
    """A collection of interactive web demos using HTML, CSS, and JavaScript."""
    
    st.title("🌐 Web Playground")
    st.markdown("Test live JavaScript browser functionalities, from camera access to AI integration.")
    st.info("💡 These demos use your browser's built-in capabilities. You may need to grant camera/microphone permissions.")

    tab_media, tab_speech, tab_ai, tab_social, tab_interactive = st.tabs([
        "📷 Media Capture", 
        "🎤 Speech & Audio", 
        "🤖 AI Integration", 
        "📱 Social Sharing",
        "🖱️ Interactive UI"
    ])

    with tab_media:
        st.header("Live Media Capture")
        st.markdown("Access your device's camera to capture photos and record videos directly in the browser.")
        media_html = """
        <div style="border:1px solid #ddd; border-radius:10px; padding:20px; margin-bottom:20px;">
            <h4>Take a Photo</h4>
            <video id="video" width="600" height="450" autoplay style="border-radius:5px;"></video>
            <canvas id="canvas" style="display:none;"></canvas>
            <div>
                <button onclick="startCamera()">Start Camera</button>
                <button onclick="takePhoto()">Take Photo</button>
                <button onclick="stopCamera()">Stop Camera</button>
            </div>
            <img id="photo" style="margin-top:10px; max-width:600px; border-radius:5px;">
        </div>
        <script>
            let stream = null;
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const photo = document.getElementById('photo');
            async function startCamera() {
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
            }
            function takePhoto() {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0);
                photo.src = canvas.toDataURL('image/png');
            }
            function stopCamera() {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    video.srcObject = null;
                }
            }
        </script>
        """
        components.html(media_html, height=650)
        
    with tab_speech:
        st.header("Live Speech & Audio")
        st.markdown("Use your microphone for speech-to-text and have your browser speak text back to you.")
        speech_html = """
        <div style="border:1px solid #ddd; border-radius:10px; padding:20px;">
            <h4>Speech-to-Text</h4>
            <button id="startSpeech" onclick="startSpeechRecognition()">Start Listening</button>
            <button id="stopSpeech" onclick="stopSpeechRecognition()" disabled>Stop Listening</button>
            <p><strong>Transcript:</strong> <span id="transcript" style="color:#28a745;"></span></p>
            <hr>
            <h4>Text-to-Speech</h4>
            <input type="text" id="textToSpeak" value="Hello from Streamlit!" style="width:70%; padding:5px;">
            <button onclick="speakText()">Speak</button>
        </div>
        <script>
            let recognition;
            if ('webkitSpeechRecognition' in window) {
                recognition = new webkitSpeechRecognition();
                recognition.continuous = true;
                recognition.interimResults = true;
                recognition.onresult = (event) => {
                    let final_transcript = '';
                    for (let i = event.resultIndex; i < event.results.length; ++i) {
                        if (event.results[i].isFinal) {
                            final_transcript += event.results[i][0].transcript;
                        }
                    }
                    document.getElementById('transcript').innerText = final_transcript;
                };
            }

            function startSpeechRecognition() {
                if (recognition) {
                    document.getElementById('transcript').innerText = '';
                    recognition.start();
                    document.getElementById('startSpeech').disabled = true;
                    document.getElementById('stopSpeech').disabled = false;
                }
            }
            function stopSpeechRecognition() {
                if (recognition) {
                    recognition.stop();
                    document.getElementById('startSpeech').disabled = false;
                    document.getElementById('stopSpeech').disabled = true;
                }
            }
            function speakText() {
                const text = document.getElementById('textToSpeak').value;
                const utterance = new SpeechSynthesisUtterance(text);
                window.speechSynthesis.speak(utterance);
            }
        </script>
        """
        components.html(speech_html, height=300)

    with tab_ai:
        st.header("Live AI Integration")
        st.markdown("Chat with the Google Gemini API directly from the browser using JavaScript.")
        # NOTE: The provided HTML for Gemini is extensive and uses a hardcoded key. 
        # For a better experience, we can simplify and allow the user to enter their own key.
        api_key_gemini = st.text_input("Enter your Google Gemini API Key", type="password")
        if api_key_gemini:
            ai_html = f"""
            <div style="border:1px solid #ddd; border-radius:10px; padding:20px;">
                <h4>Gemini AI Chat</h4>
                <textarea id="promptInput" placeholder="Enter your prompt..." style="width:100%; height:80px;"></textarea>
                <button onclick="sendToGemini()">Send Prompt</button>
                <div id="response" style="margin-top:10px; padding:10px; background-color:#f0f2f6; border-radius:5px; min-height:50px;"></div>
            </div>
            <script>
                async function sendToGemini() {{
                    const prompt = document.getElementById('promptInput').value;
                    const apiKey = "{api_key_gemini}";
                    const responseDiv = document.getElementById('response');
                    responseDiv.innerText = "Generating response...";
                    
                    const apiURL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=${{apiKey}}`;
                    
                    try {{
                        const response = await fetch(apiURL, {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{ "contents": [{{ "parts": [{{ "text": prompt }}] }}] }})
                        }});
                        if (!response.ok) {{
                            throw new Error(`API Error: ${{response.status}}`);
                        }}
                        const data = await response.json();
                        const text = data.candidates[0].content.parts[0].text;
                        responseDiv.innerText = text;
                    }} catch (error) {{
                        responseDiv.innerText = "Error: " + error.message;
                    }}
                }}
            </script>
            """
            components.html(ai_html, height=400)

    with tab_social:
        st.header("Social Media Sharing")
        st.markdown("Use JavaScript to trigger your device's native sharing capabilities.")
        social_html = """
        <div style="border:1px solid #ddd; border-radius:10px; padding:20px;">
            <input type="text" id="shareText" value="Check out this awesome Streamlit app!" style="width:100%; padding:5px; margin-bottom:10px;">
            <input type="text" id="shareUrl" value="https://streamlit.io" style="width:100%; padding:5px; margin-bottom:10px;">
            <button onclick="nativeShare()">Share Content</button>
        </div>
        <script>
            function nativeShare() {{
                if (navigator.share) {{
                    navigator.share({{
                        title: 'Shared from Streamlit',
                        text: document.getElementById('shareText').value,
                        url: document.getElementById('shareUrl').value
                    }});
                }} else {{
                    alert('Native sharing is not supported in your browser.');
                }}
            }}
        </script>
        """
        components.html(social_html, height=200)

    with tab_interactive:
        st.header("Interactive UI Elements")
        st.markdown("A simple drag-and-drop interface implemented with HTML and JavaScript.")
        drag_drop_html = """
        <style>
            #drop_zone {{ border: 2px dashed #ccc; border-radius: 10px; width: 100%; height: 200px; padding: 10px; text-align: center; }}
            #draggable {{ background-color: #28a745; color: white; padding: 10px; cursor: move; display: inline-block; border-radius: 5px; }}
            .over {{ border-color: #28a745; }}
        </style>
        <div style="border:1px solid #ddd; border-radius:10px; padding:20px;">
            <h4>Drag & Drop</h4>
            <div id="draggable" draggable="true">Drag Me</div>
            <br/><br/>
            <div id="drop_zone">Drop Zone</div>
        </div>
        <script>
            const draggable = document.getElementById('draggable');
            const dropZone = document.getElementById('drop_zone');
            
            draggable.addEventListener('dragstart', (e) => {{
                e.dataTransfer.setData('text/plain', e.target.id);
            }});
            
            dropZone.addEventListener('dragover', (e) => {{
                e.preventDefault();
                dropZone.classList.add('over');
            }});
            
            dropZone.addEventListener('dragleave', () => {{
                dropZone.classList.remove('over');
            }});
            
            dropZone.addEventListener('drop', (e) => {{
                e.preventDefault();
                dropZone.classList.remove('over');
                const id = e.dataTransfer.getData('text');
                const element = document.getElementById(id);
                dropZone.appendChild(element);
                dropZone.innerHTML += "<br>Dropped successfully!";
            }});
        </script>
        """
        components.html(drag_drop_html, height=350)

# --- App Mode 9: ML Dashboard ---
def run_ml_dashboard():
    """A comprehensive dashboard for exploring an ML workflow."""

    st.title("🚀 AI/ML Analytics Hub")
    st.markdown("An interactive dashboard to explore key stages of a machine learning project, from data cleaning to model analysis.")

    # --- Helper Functions and Data Generation (scoped to this app mode) ---
    @st.cache_data
    def create_enhanced_dataset(n_samples=150):
        # ... (This function remains the same as provided)
        np.random.seed(123)
        data = {
            'Age': np.random.gamma(2, 20, n_samples),
            'Income': np.random.lognormal(10.5, 0.4, n_samples),
            'Experience': np.random.exponential(8, n_samples),
            'Education': [random.choice(['Bachelor', 'Master', 'PhD']) for _ in range(n_samples)],
            'JobSatisfaction': np.random.beta(2, 1, n_samples) * 10,
            'Productivity': np.random.normal(78, 12, n_samples)
        }
        df = pd.DataFrame(data)
        df['Income'] = df['Income'] + df['Experience'] * 2000
        df['Productivity'] = df['Productivity'] + df['JobSatisfaction'] * 1.5
        for col in ['Age', 'Income', 'Experience']:
            missing_mask = np.random.random(n_samples) < 0.15
            df.loc[missing_mask, col] = np.nan
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].round(1)
        return df

    def display_missing_value_techniques(df):
        st.header("🔧 Advanced Missing Data Handling")
        # ... (Implementation for this section)
        
    def display_encoding_analysis(df):
        st.header("🏷️ Categorical Encoding & Feature Engineering")
        # ... (Implementation for this section)
        
    def display_model_training(df):
        st.header("⚙️ Model Training & Hyperparameter Tuning")
        # ... (Implementation for this section)

    # --- Main UI with Tabs ---
    tab_dash, tab_impute, tab_encode, tab_train = st.tabs([
        "🏠 Dashboard", "🔧 Data Imputation", "🏷️ Encoding Analysis", "⚙️ Model Training"
    ])

    df_generated = create_enhanced_dataset()

    with tab_dash:
        st.subheader("Dashboard Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{df_generated.shape[0]}")
        col2.metric("Columns", f"{df_generated.shape[1]}")
        col3.metric("Missing Values", f"{df_generated.isnull().sum().sum()}")
        st.dataframe(df_generated.head())
        
        st.subheader("Feature Distributions")
        fig = px.histogram(df_generated.dropna(), x="Income", title="Income Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with tab_impute:
        st.subheader("Handling Missing Data")
        st.write("Choose a method to fill in the missing values.")
        
        numerical_features = ['Age', 'Income', 'Experience']
        method = st.selectbox("Imputation Method", ["Mean", "Median", "KNN (k=5)", "Iterative (MICE)"])
        
        df_imputed = df_generated.copy()
        if method == "Mean":
            imputer = SimpleImputer(strategy='mean')
            df_imputed[numerical_features] = imputer.fit_transform(df_generated[numerical_features])
        elif method == "Median":
            imputer = SimpleImputer(strategy='median')
            df_imputed[numerical_features] = imputer.fit_transform(df_generated[numerical_features])
        elif method == "KNN (k=5)":
            imputer = KNNImputer(n_neighbors=5)
            df_imputed[numerical_features] = imputer.fit_transform(df_generated[numerical_features])
        elif method == "Iterative (MICE)":
            imputer = IterativeImputer(max_iter=10, random_state=0)
            df_imputed[numerical_features] = imputer.fit_transform(df_generated[numerical_features])
        
        st.write("Data after imputation:")
        st.dataframe(df_imputed.head())
        st.success(f"Missing values filled using the **{method}** strategy.")

    with tab_encode:
        st.subheader("Encoding Categorical Features")
        st.write("Convert text categories into numbers for the model.")
        
        df_clean = df_generated.dropna().copy()
        encoder = LabelEncoder()
        df_clean['Education_Encoded'] = encoder.fit_transform(df_clean['Education'])
        
        st.write("Data after Label Encoding 'Education':")
        st.dataframe(df_clean[['Education', 'Education_Encoded']].head(10))
        
    with tab_train:
        st.subheader("Train a Predictive Model")
        st.write("Train a Random Forest model to predict Productivity.")
        
        df_processed = df_generated.dropna().copy()
        df_processed['Education'] = LabelEncoder().fit_transform(df_processed['Education'])
        
        features = st.multiselect("Select Features:", list(df_processed.columns.drop('Productivity')), default=['Income', 'Experience', 'JobSatisfaction', 'Education'])
        
        if features:
            X = df_processed[features]
            y = df_processed['Productivity']
            
            n_estimators = st.slider("Number of Trees", 50, 500, 100)
            
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            model.fit(X, y)
            y_pred = model.predict(X)
            
            r2 = r2_score(y, y_pred)
            st.metric("Model R² Score", f"{r2:.3f}")

            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
            fig_imp = px.bar(importance_df.sort_values('Importance', ascending=True), x='Importance', y='Feature', orientation='h')
            st.plotly_chart(fig_imp, use_container_width=True)


# --- Main Application Router ---
st.sidebar.title("Dev Control Center ⚙️")
app_mode = st.sidebar.radio(
    "Choose Application",
    ("Web Playground","ML Dashboard", "Linear Regression", "Git Automation", "Kubernetes Dashboard", "Remote Docker Manager", "Linux Terminal Simulator", "Python Power Tools", "Gesture Docker Controller")
)

# Call the corresponding function for the selected app mode
if app_mode == "Web Playground":
    run_javascript_menu()
elif app_mode == "ML Dashboard":
    run_ml_dashboard()
elif app_mode == "Linear Regression":
    run_linear_regression()
elif app_mode == "Git Automation":
    run_git_automation()
elif app_mode == "Kubernetes Dashboard":
    run_kubernetes_dashboard()
elif app_mode == "Remote Docker Manager":
    run_remote_manager()
elif app_mode == "Linux Terminal Simulator":
    run_linux_simulator()
elif app_mode == "Python Power Tools":
    run_python_menu()
elif app_mode == "Gesture Docker Controller":
    run_gesture_controller()
# ... other elif blocks for the rest of the apps