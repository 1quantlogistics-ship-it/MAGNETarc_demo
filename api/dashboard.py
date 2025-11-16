import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

# Configuration
CONTROL_PLANE_URL = 'http://localhost:8002'
LLM_ENDPOINT = 'http://localhost:8000'

st.set_page_config(
    page_title='ARC Dashboard',
    page_icon='\U0001F916',
    layout='wide'
)

# Sidebar
st.sidebar.title('ARC Control Panel')
st.sidebar.markdown('---')

# Get system status
def get_status():
    try:
        response = requests.get(f'{CONTROL_PLANE_URL}/status', timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.sidebar.error(f'Control Plane unreachable: {e}')
        return None

status = get_status()

if status:
    # Mode display
    mode = status.get('mode', 'UNKNOWN')
    mode_colors = {'SEMI': 'orange', 'AUTO': 'blue', 'FULL': 'red'}
    st.sidebar.markdown(f'### Mode: :{mode_colors.get(mode, "gray")}[{mode}]')
    
    # System info
    st.sidebar.metric('ARC Version', status.get('arc_version', 'N/A'))
    st.sidebar.metric('Status', status.get('status', 'N/A'))
    st.sidebar.metric('Current Cycle', status.get('current_cycle', 0))
    st.sidebar.metric('Active Experiments', len(status.get('active_experiments', [])))
    
    # Mode switcher
    st.sidebar.markdown('---')
    st.sidebar.subheader('Change Mode')
    new_mode = st.sidebar.selectbox('Select Mode', ['SEMI', 'AUTO', 'FULL'])
    if st.sidebar.button('Apply Mode'):
        try:
            response = requests.post(f'{CONTROL_PLANE_URL}/mode?mode={new_mode}')
            if response.status_code == 200:
                st.sidebar.success(f'Mode changed to {new_mode}')
                time.sleep(1)
                st.rerun()
            else:
                st.sidebar.error('Mode change failed')
        except Exception as e:
            st.sidebar.error(f'Error: {e}')

# Main dashboard
st.title('ARC - Autonomous Research Collective')
st.markdown('### Multi-Agent LLM Research Platform')

if not status:
    st.error('Cannot connect to Control Plane. Please ensure the service is running.')
    st.code('python /workspace/arc/api/control_plane.py')
    st.stop()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Overview', 'Memory', 'Experiments', 'Logs', 'Execute'])

with tab1:
    st.header('System Overview')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric('Operating Mode', mode)
    with col2:
        st.metric('Current Objective', status.get('current_objective', 'N/A'))
    with col3:
        last_cycle = status.get('last_cycle')
        if last_cycle:
            st.metric('Last Cycle', datetime.fromisoformat(last_cycle).strftime('%Y-%m-%d %H:%M'))
        else:
            st.metric('Last Cycle', 'Never')
    with col4:
        st.metric('Cycle ID', status.get('current_cycle', 0))
    
    st.markdown('---')
    
    # Active experiments
    st.subheader('Active Experiments')
    active_exps = status.get('active_experiments', [])
    if active_exps:
        df = pd.DataFrame(active_exps)
        st.dataframe(df, use_container_width=True)
    else:
        st.info('No active experiments')

with tab2:
    st.header('Protocol Memory')
    
    memory_files = ['directive.json', 'history_summary.json', 'constraints.json', 'system_state.json']
    
    for fname in memory_files:
        with st.expander(f'\U0001F4C4 {fname}'):
            try:
                with open(f'/workspace/arc/memory/{fname}', 'r') as f:
                    data = json.load(f)
                st.json(data)
            except Exception as e:
                st.error(f'Error loading {fname}: {e}')

with tab3:
    st.header('Experiments')
    
    try:
        import os
        exp_dir = '/workspace/arc/experiments'
        if os.path.exists(exp_dir):
            experiments = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
            
            if experiments:
                selected_exp = st.selectbox('Select Experiment', experiments)
                
                if selected_exp:
                    exp_path = os.path.join(exp_dir, selected_exp)
                    
                    # Load experiment metadata
                    meta_path = os.path.join(exp_path, 'metadata.json')
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r') as f:
                            metadata = json.load(f)
                        st.json(metadata)
                    
                    # Load results
                    results_path = os.path.join(exp_path, 'results.json')
                    if os.path.exists(results_path):
                        with open(results_path, 'r') as f:
                            results = json.load(f)
                        
                        st.subheader('Results')
                        st.json(results)
                        
                        # Plot metrics if available
                        if 'metrics' in results:
                            metrics = results['metrics']
                            fig = go.Figure(data=[
                                go.Bar(x=list(metrics.keys()), y=list(metrics.values()))
                            ])
                            fig.update_layout(title='Experiment Metrics')
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('No experiments found')
        else:
            st.info('Experiments directory not found')
    except Exception as e:
        st.error(f'Error loading experiments: {e}')

with tab4:
    st.header('System Logs')
    
    log_type = st.selectbox('Log Type', ['Control Plane', 'Execution Logs', 'Training Logs'])
    
    try:
        if log_type == 'Control Plane':
            log_path = '/workspace/arc/logs/control_plane.log'
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    last_n = st.slider('Show last N lines', 10, 1000, 100)
                    st.text_area('Log Output', ''.join(lines[-last_n:]), height=400)
            else:
                st.info('No logs available')
        elif log_type == 'Execution Logs':
            log_dir = '/workspace/arc/logs'
            exec_logs = [f for f in os.listdir(log_dir) if f.startswith('exec_cycle_')]
            if exec_logs:
                selected_log = st.selectbox('Select Log File', exec_logs)
                with open(os.path.join(log_dir, selected_log), 'r') as f:
                    st.text_area('Log Output', f.read(), height=400)
            else:
                st.info('No execution logs available')
        else:
            st.info('Training logs not yet implemented')
    except Exception as e:
        st.error(f'Error loading logs: {e}')

with tab5:
    st.header('Execute Commands')
    
    st.warning('This tab allows executing commands through the ARC Control Plane. Commands are validated against the allowlist.')
    
    with st.form('exec_form'):
        role = st.selectbox('Role', ['Director', 'Architect', 'Critic', 'Historian', 'Executor'])
        cycle_id = st.number_input('Cycle ID', min_value=0, value=status.get('current_cycle', 0))
        command = st.text_input('Command')
        requires_approval = st.checkbox('Requires Approval', value=True)
        
        submit = st.form_submit_button('Execute')
        
        if submit and command:
            try:
                payload = {
                    'command': command,
                    'role': role,
                    'cycle_id': int(cycle_id),
                    'requires_approval': requires_approval
                }
                
                response = requests.post(f'{CONTROL_PLANE_URL}/exec', json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get('status') == 'pending_approval':
                        st.warning(result.get('message'))
                    else:
                        st.success('Command executed')
                        st.json(result)
                else:
                    st.error(f'Execution failed: {response.text}')
            except Exception as e:
                st.error(f'Error: {e}')

# Footer
st.markdown('---')
st.markdown('ARC v0.8.0 | Autonomous Research Collective')
