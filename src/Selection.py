import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Set correct MLflow server URI
mlflow.set_tracking_uri("http://localhost:5001")  # <-- Update if different
EXPERIMENT_NAME = "Iris-Experiment-Final"

def list_logged_models(client):
    print("\n📋 Available runs with F1 scores:\n")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    results = []

    if experiment is None:
        print(f"❌ Experiment '{EXPERIMENT_NAME}' not found.")
        return []

    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.f1_score DESC"])
    for r in runs:
        f1 = r.data.metrics.get("f1_score")
        if f1 is not None:
            results.append({
                "run_id": r.info.run_id,
                "f1_score": f1,
                "experiment_name": experiment.name,
                "model_name": r.data.tags.get("mlflow.runName", "Unknown")
            })

    if not results:
        print("❌ No runs with F1 score found.")
        return []

    for idx, r in enumerate(results):
        print(f"{idx+1}. {r['model_name']} | run_id: {r['run_id']} | F1: {r['f1_score']:.4f}")

    return results

def promote_model(client, run_id, model_name="iris_classifier"):
    # Step 1: Ensure model name exists in registry
    try:
        client.get_registered_model(model_name)
    except MlflowException:
        print(f"ℹ️ Registered model '{model_name}' not found. Creating...")
        client.create_registered_model(model_name)

    # Step 2: Register model using correct format
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Step 3: Promote to production
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"\n🚀 Model '{model_name}' v{result.version} promoted to PRODUCTION ✅")


if __name__ == "__main__":
    client = MlflowClient()
    models = list_logged_models(client)

    if not models:
        exit()

    print("\nChoose an option:")
    print("0. Automatically promote best model by F1 score")
    for idx, model in enumerate(models, start=1):
        print(f"{idx}. Promote manually: {model['model_name']} | run_id: {model['run_id']}")

    try:
        choice = int(input("\nEnter your choice number: "))

        if choice == 0:
            best_model = max(models, key=lambda x: x['f1_score'])
            promote_model(client, best_model['run_id'])

        elif 1 <= choice <= len(models):
            selected_model = models[choice - 1]
            promote_model(client, selected_model['run_id'])

        else:
            print("❌ Invalid choice number.")

    except ValueError:
        print("❌ Please enter a valid number.")
