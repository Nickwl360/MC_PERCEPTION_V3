#!/usr/bin/env python3
"""
Main entry point for the brain modeling and analysis pipeline.
This script provides a user interface for configuring and running different workflows.
Can be run with command-line arguments for automated execution.
"""
import os
import sys
import argparse
from workflow_handler import WorkflowManager, ConfigurationManager


def create_example_configs():
    """Create example configuration files if none exist."""
    config_manager = ConfigurationManager()
    if not config_manager.list_configurations():
        print("Creating example configuration files...")

        # Example model config
        example_model_config = {
            "settings": {
                "I": "075",
                "NR": 11,
                "NE": 26,
                "dt": 0.0002,
                "joch_data": False,
                "initial_state": [0, 0, 0, 0],
                "trajectory_duration": 8000,
                "tper": 0.03,
                "ntraj": 10000,
                "parameter_initial_guess": None
            },
            "workflow": {
                "parameters": "load",
                "trajectory": "load",
                "densities": "generate",
                "average_distributions": "generate",
                "average_trajectory": "generate",
                "generate_plots": True,
                "save_plots": True,
                "save_trajectory": True,
                "comparison": False,
                "comparison_config": "",
                "plots": {
                    "dominance": True,
                    "distributions": True,
                    "trajectories_3d": True,
                    "trajectories_topdown": True
                },
                "comparison_plots": {
                    "dominance": True,
                    "distributions": True,
                    "gamma_lower": True,
                    "ac_ad": True,
                    "trajectories_3d": True,
                    "trajectories_topdown": True
                }
            }
        }
        config_manager.save_configuration(example_model_config, "example_model.json")

        # Example comparison config
        example_comparison_config = {
            "settings": {
                "I": "100",
                "NR": 11,
                "NE": 26,
                "dt": 0.0002,
                "joch_data": True,
                "initial_state": [0, 0, 0, 0],
                "trajectory_duration": 8000,
                "tper": 0.03,
                "ntraj": 10000,
                "parameter_initial_guess": None
            },
            "workflow": {
                "parameters": "load",
                "trajectory": "load",
                "densities": "generate",
                "average_distributions": "generate",
                "average_trajectory": "generate",
                "generate_plots": True,
                "save_plots": True,
                "save_trajectory": True
            }
        }
        config_manager.save_configuration(example_comparison_config, "example_comparison.json")

        # Example batch config with comparison
        example_batch_config = {
            "settings": {
                "I": "075",
                "NR": 11,
                "NE": 26,
                "dt": 0.0002,
                "joch_data": False,
                "initial_state": [0, 0, 0, 0],
                "trajectory_duration": 8000,
                "tper": 0.03,
                "ntraj": 10000,
                "parameter_initial_guess": None
            },
            "workflow": {
                "parameters": "load",
                "trajectory": "load",
                "densities": "generate",
                "average_distributions": "generate",
                "average_trajectory": "generate",
                "generate_plots": True,
                "save_plots": True,
                "save_trajectory": True,
                "comparison": True,
                "comparison_config": "example_comparison.json",
                "plots": {
                    "dominance": True,
                    "distributions": True,
                    "trajectories_3d": True,
                    "trajectories_topdown": True
                },
                "comparison_plots": {
                    "dominance": True,
                    "distributions": False,
                    "gamma_lower": True,
                    "ac_ad": False,
                    "trajectories_3d": True,
                    "trajectories_topdown": False
                }
            }
        }
        config_manager.save_configuration(example_batch_config, "example_batch.json")

        print("Example configuration files created successfully.")


def display_main_menu():
    """Display the main menu options."""
    print("\n======== Brain Modeling Pipeline ========")
    print("1. Run a single workflow")
    print("2. Run a batch of workflows")
    print("3. Create a new configuration")
    print("4. Edit an existing configuration")
    print("5. View available configurations")
    print("6. Exit")
    return input("Enter your choice (1-6): ")


def create_new_configuration(config_manager):
    """Guide the user through creating a new configuration."""
    print("\n======== Create New Configuration ========")

    # Settings
    settings = {}
    settings["I"] = input("Enter I value: ")
    settings["NR"] = int(input("Enter NR value: "))
    settings["NE"] = int(input("Enter NE value: "))
    settings["dt"] = float(input("Enter dt value: "))
    settings["joch_data"] = input("Use Joch data? (true/false): ").strip().lower() == "true"
    settings["initial_state"] = [float(x) for x in
                                 input("Enter initial state (comma-separated, e.g., 0,0,0,0): ").split(',')]
    settings["trajectory_duration"] = float(input("Enter trajectory duration: "))
    settings["tper"] = float(input("Enter Tper value for average trajectory (default: 0.03): ") or "0.03")
    settings["ntraj"] = int(input("Enter Ntraj value for average trajectory (default: 10000): ") or "10000")
    param_guess = input("Enter parameter initial guess (optional, comma-separated): ").strip()
    settings["parameter_initial_guess"] = [float(x) for x in param_guess.split(',')] if param_guess else None

    # Workflow
    workflow = {}
    workflow["parameters"] = input("Find parameters? (infer/load/none): ").strip().lower()
    workflow["trajectory"] = input("Generate or load trajectory? (generate/load/none): ").strip().lower()
    workflow["densities"] = input("Generate or load density? (generate/load/none): ").strip().lower()
    workflow["average_distributions"] = input(
        "Generate or load average distributions? (generate/load/none): ").strip().lower()
    workflow["average_trajectory"] = input(
        "Generate or load average trajectory flows? (generate/load/none): ").strip().lower()

    workflow["generate_plots"] = input("Generate plots? (yes/no): ").strip().lower() == "yes"
    if workflow["generate_plots"] == "yes":
        workflow["save_plots"] = input("Save plots? (yes/no): ").strip().lower() == "yes"
    else: workflow["save_plots"] = False

    workflow["save_trajectory"] = input("Save trajectory? (yes/no): ").strip().lower() == "yes"
    workflow["comparison"] = input("Perform comparison with another model? (yes/no): ").strip().lower() == "yes"

    if workflow["comparison"]:
        workflow["comparison_config"] = input("Comparison configuration file: ").strip()
    else:
        workflow["comparison_config"] = ""

    # Plot options
    plots = {}
    plots["dominance"] = input("Generate dominance statistics plots? (yes/no): ").strip().lower() == "yes"
    plots["distributions"] = input("Generate distribution plots? (yes/no): ").strip().lower() == "yes"
    plots["trajectories_3d"] = input("Generate 3D trajectory plots? (yes/no): ").strip().lower() == "yes"
    plots["trajectories_topdown"] = input("Generate top-down trajectory plots? (yes/no): ").strip().lower() == "yes"
    workflow["plots"] = plots

    # Comparison plot options
    if workflow["comparison"]:
        comparison_plots = {}
        comparison_plots["dominance"] = input("Compare dominance statistics? (yes/no): ").strip().lower() == "yes"
        comparison_plots["distributions"] = input("Compare distributions? (yes/no): ").strip().lower() == "yes"
        comparison_plots["gamma_lower"] = input(
            "Compare gamma and lower distributions? (yes/no): ").strip().lower() == "yes"
        comparison_plots["ac_ad"] = input("Compare AC and AD distributions? (yes/no): ").strip().lower() == "yes"
        comparison_plots["trajectories_3d"] = input("Compare 3D trajectories? (yes/no): ").strip().lower() == "yes"
        comparison_plots["trajectories_topdown"] = input(
            "Compare top-down trajectories? (yes/no): ").strip().lower() == "yes"
        workflow["comparison_plots"] = comparison_plots

    # Save configuration
    filename = input("Enter filename to save this configuration (without .json extension): ").strip() + ".json"
    config = {"settings": settings, "workflow": workflow}
    config_manager.save_configuration(config, filename)
    print(f"Configuration saved as {filename}")


def edit_configuration(config_manager):
    """Allow the user to edit an existing configuration."""
    print("\n======== Edit Configuration ========")
    config_list = config_manager.list_configurations()

    if not config_list:
        print("No configurations found. Please create a new one.")
        return

    for index, config in enumerate(config_list):
        print(f"{index}: {config}")

    choice = input("Enter the index of the configuration to edit: ")
    try:
        index = int(choice)
        filename = config_list[index]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return

    config = config_manager.load_configuration(filename)

    # Display and edit settings
    print("\nCurrent settings:")
    for key, value in config["settings"].items():
        print(f"  {key}: {value}")

    # Ask which settings to edit
    print("\nEnter new values (press Enter to keep current value):")
    for key in config["settings"]:
        new_value = input(f"{key} [{config['settings'][key]}]: ").strip()
        if new_value:
            if key in ["NR", "NE", "ntraj"]:
                config["settings"][key] = int(new_value)
            elif key in ["dt", "trajectory_duration", "tper"]:
                config["settings"][key] = float(new_value)
            elif key == "joch_data":
                config["settings"][key] = new_value.lower() == "true"
            elif key == "initial_state":
                config["settings"][key] = [float(x) for x in new_value.split(',')]
            elif key == "parameter_initial_guess":
                config["settings"][key] = [float(x) for x in new_value.split(',')] if new_value else None
            else:
                config["settings"][key] = new_value

    # Display and edit workflow options
    print("\nCurrent workflow options:")
    for key, value in config["workflow"].items():
        if key not in ["plots", "comparison_plots"]:
            print(f"  {key}: {value}")

    # Ask which workflow options to edit
    print("\nEnter new values (press Enter to keep current value):")
    for key in list(config["workflow"].keys()):
        if key not in ["plots", "comparison_plots"]:
            if isinstance(config["workflow"][key], bool):
                new_value = input(f"{key} [{config['workflow'][key]}]: ").strip()
                if new_value:
                    config["workflow"][key] = new_value.lower() == "yes" or new_value.lower() == "true"
            else:
                new_value = input(f"{key} [{config['workflow'][key]}]: ").strip()
                if new_value:
                    config["workflow"][key] = new_value

    # Edit plot options
    if "plots" in config["workflow"]:
        print("\nCurrent plot options:")
        for key, value in config["workflow"]["plots"].items():
            print(f"  {key}: {value}")

        print("\nEnter new values (press Enter to keep current value):")
        for key in config["workflow"]["plots"]:
            new_value = input(f"{key} [{config['workflow']['plots'][key]}]: ").strip()
            if new_value:
                config["workflow"]["plots"][key] = new_value.lower() == "yes" or new_value.lower() == "true"

    # Edit comparison plot options if comparison is enabled
    if config["workflow"].get("comparison", False) and "comparison_plots" in config["workflow"]:
        print("\nCurrent comparison plot options:")
        for key, value in config["workflow"]["comparison_plots"].items():
            print(f"  {key}: {value}")

        print("\nEnter new values (press Enter to keep current value):")
        for key in config["workflow"]["comparison_plots"]:
            new_value = input(f"{key} [{config['workflow']['comparison_plots'][key]}]: ").strip()
            if new_value:
                config["workflow"]["comparison_plots"][key] = new_value.lower() == "yes" or new_value.lower() == "true"

    # Save updated configuration
    config_manager.save_configuration(config, filename)
    print(f"Configuration updated and saved as {filename}")


def view_configurations(config_manager):
    """Display available configurations and their details."""
    print("\n======== Available Configurations ========")
    config_list = config_manager.list_configurations()

    if not config_list:
        print("No configurations found.")
        return

    for index, config_file in enumerate(config_list):
        print(f"\n{index}: {config_file}")
        try:
            config = config_manager.load_configuration(config_file)
            print("  Settings:")
            for key, value in config["settings"].items():
                print(f"    {key}: {value}")
            print("  Workflow:")
            for key, value in config["workflow"].items():
                if key not in ["plots", "comparison_plots"]:
                    print(f"    {key}: {value}")

            if "plots" in config["workflow"]:
                print("  Plot Options:")
                for key, value in config["workflow"]["plots"].items():
                    print(f"    {key}: {value}")

            if "comparison_plots" in config["workflow"] and config["workflow"].get("comparison", False):
                print("  Comparison Plot Options:")
                for key, value in config["workflow"]["comparison_plots"].items():
                    print(f"    {key}: {value}")
        except Exception as e:
            print(f"  Error reading configuration: {e}")

    input("\nPress Enter to continue...")


def run_single_workflow(workflow_manager, config_manager):
    """Run a single workflow with user-selected configuration."""
    print("\n======== Run Single Workflow ========")
    config_list = config_manager.list_configurations()

    if not config_list:
        print("No configurations found. Please create a new one.")
        return

    for index, config in enumerate(config_list):
        print(f"{index}: {config}")

    choice = input("Enter the index of the configuration to run: ")
    try:
        index = int(choice)
        filename = config_list[index]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return

    print(f"Running workflow with configuration: {filename}")
    workflow_manager.initialize(filename)
    workflow_manager.run_workflow()


def run_batch_workflows(workflow_manager, config_manager):
    """Run multiple workflows in batch mode."""
    print("\n======== Run Batch Workflows ========")
    config_list = config_manager.list_configurations()

    if not config_list:
        print("No configurations found. Please create a new one.")
        return

    print("Available configurations:")
    for index, config in enumerate(config_list):
        print(f"{index}: {config}")

    indices = input("Enter indices of configurations to run (comma-separated): ").strip()
    batch_files = []

    try:
        for idx in indices.split(','):
            index = int(idx.strip())
            batch_files.append(config_list[index])
    except (ValueError, IndexError):
        print("Invalid indices provided.")
        return

    print(f"Running batch workflow with {len(batch_files)} configurations:")
    for file in batch_files:
        print(f"  - {file}")

    confirm = input("Proceed? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Batch run cancelled.")
        return

    workflow_manager.run_batch(batch_files)


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Brain Modeling Pipeline')
    parser.add_argument('--auto', action='store_true', help='Run in automatic mode without prompts')
    parser.add_argument('--config', nargs='+', help='Configuration file(s) to use in automatic mode')
    parser.add_argument('--create-examples', action='store_true', help='Create example configuration files')

    args = parser.parse_args()

    # Create example configurations if requested
    if args.create_examples:
        create_example_configs()
        if not args.auto:
            return

    # Create managers
    config_manager = ConfigurationManager()
    workflow_manager = WorkflowManager()

    # Check if any configuration files exist, if not create examples
    if not config_manager.list_configurations():
        print("No configuration files found. Creating examples...")
        create_example_configs()

    # Automatic mode
    if args.auto:
        if not args.config:
            print("Error: Automatic mode requires at least one configuration file.")
            print("Use --config config1.json [config2.json ...] to specify configurations.")
            return

        print(f"Running in automatic mode with {len(args.config)} configuration(s).")
        workflow_manager.run_batch(args.config)
        return

    # Interactive mode
    while True:
        choice = display_main_menu()

        if choice == '1':
            run_single_workflow(workflow_manager, config_manager)
        elif choice == '2':
            run_batch_workflows(workflow_manager, config_manager)
        elif choice == '3':
            create_new_configuration(config_manager)
        elif choice == '4':
            edit_configuration(config_manager)
        elif choice == '5':
            view_configurations(config_manager)
        elif choice == '6':
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()