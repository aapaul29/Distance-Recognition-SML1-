from utils import load_config, load_dataset, load_test_dataset, print_results, save_results

# sklearn imports...
# SVRs are not allowed in this project.

if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    # Load dataset: images and corresponding minimum distance values
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    # TODO: Your implementation starts here

    # possible preprocessing steps ... training the model

    # Evaluation
    # print_results(gt, pred)

    # Save the results
    # save_results(test_pred)