import joblib

# Load the saved pipeline
pipeline = joblib.load("Fraud_detection_pipeline.pkl")

print("Pipeline Loaded Successfully!\n")

print("Pipeline:")
print(pipeline)

print("\nFeature Names:")

if hasattr(pipeline, "feature_names_in_"):
    print(pipeline.feature_names_in_)
else:
    print("feature_names_in_ not found")

print("\nPipeline Steps:")
print(pipeline.named_steps)