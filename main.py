from sign_language_model import SignLanguageModel

# Create instance of SignLanguageModel
model = SignLanguageModel()

# Load data
x_train, y_train, x_test, y_test = model.load_data()

# Build model
model.build_model()

# Train model
hist = model.train_model(x_train, model.y_train_OH)

# Evaluate model
test_loss, test_acc = model.evaluate_model(x_test, model.y_test_OH)
print(f"Test accuracy: {test_acc}")

# Predict and plot mislabeled examples
y_preds = model.predict(x_test)
model.plot_mislabeled_examples(x_test, y_test, y_preds)
