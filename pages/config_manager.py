import streamlit as st
import json
import pandas as pd
import os
from datetime import datetime

# Initialize session state
def init_session_state():
    if 'config' not in st.session_state:
        st.session_state.config = {
            "variables": [],
            "intermediates": [],
            "objectives": []  # Remove the top-level expressions key
        }
    if 'edit_var_index' not in st.session_state:
        st.session_state.edit_var_index = None
    if 'edit_intermediate_index' not in st.session_state:
        st.session_state.edit_intermediate_index = None
    if 'edit_obj_index' not in st.session_state:
        st.session_state.edit_obj_index = None
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Variables"

# Save configuration to file
def save_config():
    with open('optimization_config.json', 'w') as f:
        json.dump(st.session_state.config, f, indent=4)
    st.success("Configuration saved successfully!")

# Load configuration (including old data migration logic)
def load_config():
    if os.path.exists('optimization_config.json'):
        with open('optimization_config.json', 'r') as f:
            loaded_config = json.load(f)
            
            # Ensure necessary keys exist (compatible with old configurations)
            for key in ["intermediates", "variables", "objectives"]:
                if key not in loaded_config:
                    loaded_config[key] = []
            
            st.session_state.config = loaded_config
        st.success("Configuration loaded successfully!")
    else:
        st.warning("Configuration file not found, using default config")
        st.session_state.config = {
            "variables": [],
            "intermediates": [],
            "objectives": []
        }

# Add variable (unchanged)
def add_variable(name, vtype, lb=None, ub=None, categories=None, description=""):
    new_var = {
        "name": name, 
        "type": vtype,
        "description": description
    }
    
    if vtype == "float":
        new_var["lb"] = float(lb)
        new_var["ub"] = float(ub)
    elif vtype == "int":
        new_var["lb"] = int(lb)
        new_var["ub"] = int(ub)
    elif vtype == "categories":
        new_var["categories"] = [c.strip() for c in categories.split(',')]
    
    st.session_state.config["variables"].append(new_var)
    st.session_state.edit_var_index = None

# Update variable (unchanged)
def update_variable(index, name, vtype, lb=None, ub=None, categories=None, description=""):
    updated_var = {
        "name": name, 
        "type": vtype,
        "description": description
    }
    
    if vtype == "float":
        updated_var["lb"] = float(lb)
        updated_var["ub"] = float(ub)
    elif vtype == "int":
        updated_var["lb"] = int(lb)
        updated_var["ub"] = int(ub)
    elif vtype == "categories":
        updated_var["categories"] = [c.strip() for c in categories.split(',')]
    
    st.session_state.config["variables"][index] = updated_var
    st.session_state.edit_var_index = None

# Delete variable (unchanged)
def delete_variable(index):
    var_name = st.session_state.config["variables"][index]["name"]
    
    # Remove references in expressions (now expressions are in objectives)
    for obj in st.session_state.config["objectives"]:
        if var_name in obj.get("expression", ""):
            st.warning(f"Warning: Variable '{var_name}' is used in expressions. Deleting it may invalidate expressions.")
    
    del st.session_state.config["variables"][index]
    st.session_state.edit_var_index = None

# Add intermediate variable (unchanged)
def add_intermediate_var(name, expression, description=""):
    new_var = {
        "name": name,
        "expression": expression,
        "description": description
    }
    st.session_state.config["intermediates"].append(new_var)
    st.session_state.edit_intermediate_index = None

# Update intermediate variable (unchanged)
def update_intermediate_var(index, name, expression, description=""):
    updated_var = {
        "name": name,
        "expression": expression,
        "description": description
    }
    st.session_state.config["intermediates"][index] = updated_var
    st.session_state.edit_intermediate_index = None

# Delete intermediate variable (unchanged)
def delete_intermediate_var(index):
    del st.session_state.config["intermediates"][index]
    st.session_state.edit_intermediate_index = None

# Add objective (modified: add the expression field)
def add_objective(name, target, description=""):
    new_obj = {
        "name": name, 
        "target": target,
        "description": description,
        "expression": ""  # Initialize the expression field
    }
    st.session_state.config["objectives"].append(new_obj)
    st.session_state.edit_obj_index = None

# Update objective (modified: retain the expression field)
def update_objective(index, name, target, description=""):
    old_obj = st.session_state.config["objectives"][index]
    new_obj = {
        "name": name, 
        "target": target,
        "description": description,
        "expression": old_obj.get("expression", "")  # Retain the original expression
    }
    
    st.session_state.config["objectives"][index] = new_obj
    st.session_state.edit_obj_index = None

# Delete objective (modified: no need to operate on expressions)
def delete_objective(index):
    del st.session_state.config["objectives"][index]
    st.session_state.edit_obj_index = None

# Main application (key modifications to the Expressions/Preview/Configuration tab pages)
def main():
    st.set_page_config(
        page_title="Multi-Objective Optimization Configurator",
        page_icon=":gear:",
        layout="wide"
    )
    
    st.title("📊 Multi-Objective Optimization Configurator")
    st.markdown("Manage your optimization variables, objectives, and function relationships")
    
    init_session_state()
    
    # Sidebar (unchanged)
    with st.sidebar:
        st.header("System Operations")
        tabs = ["Variables", "Intermediate Variables", "Objectives", "Expressions", "Preview", "Configuration"]
        st.selectbox("Navigation", tabs, key="active_tab", index=tabs.index(st.session_state.active_tab))
        
        st.divider()
        
        st.subheader("Configuration Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Config", use_container_width=True):
                save_config()
        with col2:
            if st.button("📥 Load Config", use_container_width=True):
                load_config()
        
        st.download_button(
            label="⬇️ Export Config",
            data=json.dumps(st.session_state.config, indent=4),
            file_name=f"optimization_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
        
        uploaded_file = st.file_uploader("Import Config", type=["json"])
        if uploaded_file is not None:
            try:
                loaded_config = json.load(uploaded_file)
                
                # Compatibility with old configuration migration
                for key in ["intermediates", "variables", "objectives"]:
                    if key not in loaded_config:
                        loaded_config[key] = []
                old_expressions = loaded_config.get("expressions", {})
                for obj in loaded_config["objectives"]:
                    obj_name = obj.get("name")
                    obj["expression"] = old_expressions.get(obj_name, "")
                if "expressions" in loaded_config:
                    del loaded_config["expressions"]
                
                st.session_state.config = loaded_config
                st.success("Configuration imported successfully!")
            except Exception as e:
                st.error(f"Import failed: {str(e)}")
        
        st.divider()
        st.caption("Developed by PS/PCB-AP")

    # Variables management (unchanged)
    if st.session_state.active_tab == "Variables":
        st.header("Variables Management")
        st.write("Define optimization variables and their boundaries")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            with st.expander("Add New Variable", expanded=True):
                with st.form("add_variable_form"):
                    var_name = st.text_input("Variable Name", key="new_var_name")
                    var_type = st.selectbox("Variable Type", ["float", "int", "categories"], key="new_var_type")
                    
                    var_desc = st.text_input("Description", key="new_var_desc", 
                                           help="Optional description for this variable")
                    
                    if var_type in ["float", "int"]:
                        col_lb, col_ub = st.columns(2)
                        with col_lb:
                            lb = st.number_input("Lower Bound", key="new_var_lb", value=0.0)
                        with col_ub:
                            ub = st.number_input("Upper Bound", key="new_var_ub", value=10.0)
                    else:
                        categories = st.text_input("Categories (comma separated)", key="new_var_categories", value="A,B,C")
                    
                    if st.form_submit_button("Add Variable"):
                        if var_name:
                            if var_type in ["float", "int"]:
                                add_variable(var_name, var_type, lb, ub, description=var_desc)
                            else:
                                add_variable(var_name, var_type, categories=categories, description=var_desc)
                        else:
                            st.warning("Please enter a variable name")
        
        with col1:
            if st.session_state.config["variables"]:
                st.subheader("Current Variables")
                var_data = []
                for i, var in enumerate(st.session_state.config["variables"]):
                    row = {
                        "ID": i+1,
                        "Name": var["name"],
                        "Type": var["type"],
                        "Description": var.get("description", "")
                    }
                    
                    if var["type"] in ["float", "int"]:
                        row["Lower Bound"] = var.get("lb", "")
                        row["Upper Bound"] = var.get("ub", "")
                    else:
                        row["Categories"] = ", ".join(var.get("categories", []))
                    
                    var_data.append(row)
                
                df = pd.DataFrame(var_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No variables added yet")
    
        # Edit/Delete variables (unchanged)
        if st.session_state.config["variables"]:
            st.subheader("Edit Variables")
            var_names = [var["name"] for var in st.session_state.config["variables"]]
            selected_var = st.selectbox("Select variable to edit", var_names, index=st.session_state.edit_var_index or 0)
            var_index = var_names.index(selected_var)
            var = st.session_state.config["variables"][var_index]
            
            with st.form(f"edit_var_{var_index}"):
                edit_name = st.text_input("Variable Name", value=var["name"])
                edit_type = st.selectbox("Variable Type", ["float", "int", "categories"], 
                                        index=["float", "int", "categories"].index(var["type"]))
                
                edit_desc = st.text_input("Description", value=var.get("description", ""),
                                        help="Optional description for this variable")
                
                if edit_type in ["float", "int"]:
                    col_lb, col_ub = st.columns(2)
                    with col_lb:
                        edit_lb = st.number_input("Lower Bound", value=var.get("lb", 0.0))
                    with col_ub:
                        edit_ub = st.number_input("Upper Bound", value=var.get("ub", 10.0))
                else:
                    edit_categories = st.text_input("Categories (comma separated)", 
                                                   value=", ".join(var.get("categories", [])))
                
                col1, col2, col3 = st.columns([1,1,2])
                with col1:
                    if st.form_submit_button("Update"):
                        if edit_type in ["float", "int"]:
                            update_variable(var_index, edit_name, edit_type, edit_lb, edit_ub, description=edit_desc)
                        else:
                            update_variable(var_index, edit_name, edit_type, categories=edit_categories, description=edit_desc)
                with col2:
                    if st.form_submit_button("Delete"):
                        delete_variable(var_index)
                with col3:
                    if st.form_submit_button("Cancel"):
                        st.session_state.edit_var_index = None

    # Intermediate Variables management (unchanged)
    elif st.session_state.active_tab == "Intermediate Variables":
        st.header("Intermediate Variables")
        st.write("Define intermediate calculation variables")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            with st.expander("Add Intermediate Variable", expanded=True):
                with st.form("add_intermediate_form"):
                    var_name = st.text_input("Variable Name", key="new_inter_name")
                    var_expr = st.text_area("Expression", height=100, key="new_inter_expr",
                                          help="Mathematical expression using variables and other intermediates")
                    var_desc = st.text_input("Description", key="new_inter_desc")
                    
                    if st.form_submit_button("Add Variable"):
                        if var_name and var_expr:
                            add_intermediate_var(var_name, var_expr, description=var_desc)
                        else:
                            st.warning("Name and expression are required!")
        
        with col1:
            if st.session_state.config["intermediates"]:
                st.subheader("Current Intermediate Variables")
                var_data = []
                for i, var in enumerate(st.session_state.config["intermediates"]):
                    var_data.append({
                        "ID": i+1,
                        "Name": var["name"],
                        "Expression": var["expression"],
                        "Description": var.get("description", "")
                    })
                
                st.dataframe(pd.DataFrame(var_data), use_container_width=True, hide_index=True)
            else:
                st.info("No intermediate variables added yet")
    
        # Edit/Delete intermediates (unchanged)
        if st.session_state.config["intermediates"]:
            st.subheader("Edit Intermediate Variables")
            var_names = [var["name"] for var in st.session_state.config["intermediates"]]
            selected_var = st.selectbox("Select variable to edit", var_names, 
                                       index=st.session_state.edit_intermediate_index or 0)
            var_index = var_names.index(selected_var)
            var = st.session_state.config["intermediates"][var_index]
            
            with st.form(f"edit_inter_{var_index}"):
                edit_name = st.text_input("Variable Name", value=var["name"])
                edit_expr = st.text_area("Expression", value=var["expression"], height=100)
                edit_desc = st.text_input("Description", value=var.get("description", ""))
                
                col1, col2, col3 = st.columns([1,1,2])
                with col1:
                    if st.form_submit_button("Update"):
                        update_intermediate_var(var_index, edit_name, edit_expr, description=edit_desc)
                with col2:
                    if st.form_submit_button("Delete"):
                        delete_intermediate_var(var_index)
                with col3:
                    if st.form_submit_button("Cancel"):
                        st.session_state.edit_intermediate_index = None

    # Objectives management (unchanged)
    elif st.session_state.active_tab == "Objectives":
        st.header("Objectives Management")
        st.write("Define optimization objectives and their targets")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            with st.expander("Add New Objective", expanded=True):
                with st.form("add_objective_form"):
                    obj_name = st.text_input("Objective Name", key="new_obj_name")
                    
                    obj_desc = st.text_input("Description", key="new_obj_desc", 
                                           help="Optional description for this objective")
                    
                    obj_target = st.selectbox("Optimization Target", ["minimize", "maximize"], key="new_obj_target")
                    
                    if st.form_submit_button("Add Objective"):
                        if obj_name:
                            add_objective(obj_name, obj_target, description=obj_desc)
                        else:
                            st.warning("Please enter an objective name")
        
        with col1:
            if st.session_state.config["objectives"]:
                st.subheader("Current Objectives")
                obj_data = []
                for i, obj in enumerate(st.session_state.config["objectives"]):
                    obj_data.append({
                        "ID": i+1,
                        "Name": obj["name"],
                        "Target": obj["target"],
                        "Description": obj.get("description", "")
                    })
                
                df = pd.DataFrame(obj_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No objectives added yet")
    
        # Edit/Delete objectives (unchanged)
        if st.session_state.config["objectives"]:
            st.subheader("Edit Objectives")
            obj_names = [obj["name"] for obj in st.session_state.config["objectives"]]
            selected_obj = st.selectbox("Select objective to edit", obj_names, index=st.session_state.edit_obj_index or 0)
            obj_index = obj_names.index(selected_obj)
            obj = st.session_state.config["objectives"][obj_index]
            
            with st.form(f"edit_obj_{obj_index}"):
                edit_name = st.text_input("Objective Name", value=obj["name"])
                
                edit_desc = st.text_input("Description", value=obj.get("description", ""),
                                        help="Optional description for this objective")
                
                edit_target = st.selectbox("Optimization Target", ["minimize", "maximize"], 
                                          index=0 if obj["target"] == "minimize" else 1)
                
                col1, col2, col3 = st.columns([1,1,2])
                with col1:
                    if st.form_submit_button("Update"):
                        update_objective(obj_index, edit_name, edit_target, description=edit_desc)
                with col2:
                    if st.form_submit_button("Delete"):
                        delete_objective(obj_index)
                with col3:
                    if st.form_submit_button("Cancel"):
                        st.session_state.edit_obj_index = None

    # Expressions management (core modification: adjust the expression storage location)
    elif st.session_state.active_tab == "Expressions":
        st.header("Function Relationships")
        st.write("Define mathematical relationships between variables, intermediates, and objectives")
        
        # Display available intermediate variables (unchanged)
        if st.session_state.config["intermediates"]:
            st.subheader("Available Intermediate Variables")
            inter_data = []
            for var in st.session_state.config["intermediates"]:
                inter_data.append({
                    "Name": var["name"],
                    "Expression": var["expression"],
                    "Description": var.get("description", "")
                })
            st.dataframe(pd.DataFrame(inter_data), use_container_width=True)
        
        if not st.session_state.config["objectives"]:
            st.warning("Please add objectives first")
        else:
            for idx, obj in enumerate(st.session_state.config["objectives"]):
                obj_name = obj["name"]
                obj_target = obj["target"]
                
                with st.expander(f"Objective: {obj_name} ({obj_target})", expanded=True):
                    if obj.get("description"):
                        st.caption(f"Description: {obj['description']}")
                    
                    variables = [var["name"] for var in st.session_state.config["variables"]]
                    st.caption(f"Available variables: {', '.join(variables)}")
                    
                    # Read the expression from the objective
                    current_expr = obj.get("expression", "")
                    new_expr = st.text_area(
                        f"Define expression for {obj_name}", 
                        value=current_expr,
                        height=100,
                        key=f"expr_{obj_name}"
                    )
                    
                    if st.button("Save Expression", key=f"save_{obj_name}"):
                        # Directly update the expression field of the objective
                        st.session_state.config["objectives"][idx]["expression"] = new_expr
                        st.success(f"Expression for {obj_name} saved")
                    
                    if new_expr:
                        st.markdown("**Expression Preview**")
                        st.code(f"{obj_name} = {new_expr}")

    # Configuration preview (core modification: read expressions from objectives)
    elif st.session_state.active_tab == "Preview":
        st.header("Configuration Preview")
        st.write("Complete overview of the current optimization configuration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Variables")
            if st.session_state.config["variables"]:
                var_data = []
                for i, var in enumerate(st.session_state.config["variables"]):
                    row = {
                        "ID": i+1,
                        "Name": var["name"],
                        "Type": var["type"],
                        "Description": var.get("description", "")
                    }
                    
                    if var["type"] in ["float", "int"]:
                        row["Lower Bound"] = var.get("lb", "")
                        row["Upper Bound"] = var.get("ub", "")
                    else:
                        row["Categories"] = ", ".join(var.get("categories", []))
                    
                    var_data.append(row)
                
                st.dataframe(pd.DataFrame(var_data), use_container_width=True, hide_index=True)
            else:
                st.info("No variables added yet")
            
            st.subheader("Intermediate Variables")
            if st.session_state.config["intermediates"]:
                inter_data = []
                for i, var in enumerate(st.session_state.config["intermediates"]):
                    inter_data.append({
                        "ID": i+1,
                        "Name": var["name"],
                        "Expression": var["expression"],
                        "Description": var.get("description", "")
                    })
                st.dataframe(pd.DataFrame(inter_data), use_container_width=True)
            else:
                st.info("No intermediate variables defined")

            st.subheader("Objectives")
            if st.session_state.config["objectives"]:
                obj_data = []
                for i, obj in enumerate(st.session_state.config["objectives"]):
                    obj_data.append({
                        "ID": i+1,
                        "Name": obj["name"],
                        "Target": obj["target"],
                        "Description": obj.get("description", "")
                    })
                
                st.dataframe(pd.DataFrame(obj_data), use_container_width=True, hide_index=True)
            else:
                st.info("No objectives added yet")


        with col2:
            st.subheader("Function Relationships")
            if st.session_state.config["objectives"]:
                for obj in st.session_state.config["objectives"]:
                    obj_name = obj["name"]
                    expr = obj.get("expression", "")
                    if expr:
                        st.markdown(f"**{obj_name}** = `{expr}`")
                    else:
                        st.markdown(f"**{obj_name}**: *expression not defined*")
                if not any(obj.get("expression", "").strip() for obj in st.session_state.config["objectives"]):
                    st.info("No function relationships defined")
            else:
                st.info("No objectives added yet")
            
            st.subheader("Configuration Summary")
            summary = f"""
            - **Variables Count**: {len(st.session_state.config["variables"])}
            - **Intermediate Variables**: {len(st.session_state.config["intermediates"])}
            - **Objectives Count**: {len(st.session_state.config["objectives"])}
            - **Expressions Defined**: {sum(1 for obj in st.session_state.config["objectives"] if obj.get("expression", "").strip())}
            """
            st.markdown(summary)

    # Configuration details (modified: remove the display of top-level expressions)
    elif st.session_state.active_tab == "Configuration":
        st.header("Configuration Details")
        st.write("Complete JSON configuration data")
        
        st.json(st.session_state.config)
        
        st.download_button(
            label="Download Full Configuration",
            data=json.dumps(st.session_state.config, indent=4),
            file_name="optimization_config.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()