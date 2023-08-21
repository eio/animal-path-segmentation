import os


def parse_accuracy(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        # Accuracy score is on Line 5 of "model_evaluation.txt"
        if len(lines) >= 5:
            # Line 5 is at index 4 (0-based index)
            return lines[4].strip()
        else:
            return None


def search_and_parse_accuracy(directory):
    results = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file == "model_evaluation.txt":
                file_path = os.path.join(root, file)
                result = parse_accuracy(file_path)
                if result is not None:
                    identity = file_path.split("/")[-3]
                    accuracy = result.split()[-1]
                    results.append((identity, float(accuracy)))
    print()
    results = sorted(results, key=lambda x: x[1], reverse=True)
    max_identity_length = max(len(identity) for identity, _ in results)
    print("------------------------------------------------------------------")
    print("           Model Parameters : Test Accuracy % (Ordered)")
    print("------------------------------------------------------------------")
    for identity, accuracy in results:
        accuracy = accuracy * 100
        print(f"{identity: <{max_identity_length}}   :   {accuracy:.3f} %")
        print("__________________________________________________________________")
    print()


# Call the function with the current working directory
search_and_parse_accuracy(os.getcwd())
