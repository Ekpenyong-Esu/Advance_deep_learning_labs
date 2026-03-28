from wandb_logger import WandbLogger
import random
import time


def main():
    # Initialize logger
    logger = WandbLogger(
        project="wandb-test",
        run_name="logger_test_run",
        config={
            "epochs": 5,
            "learning_rate": 0.001,
            "test_type": "integration"
        },
        mode="online"   # change to "offline" if needed
    )

    print("Logger initialized!")

    # Simulate training loop
    for epoch in range(5):
        loss = random.uniform(0.5, 1.5)
        accuracy = random.uniform(50, 100)

        # Log using your custom logger
        logger.log_metrics(
            epoch=epoch,
            loss=loss,
            accuracy=accuracy
        )

        print(f"Epoch {epoch}: loss={loss:.3f}, acc={accuracy:.2f}")
        time.sleep(1)

    # Test scalar logging
    logger.log_scalar("final_accuracy", accuracy)

    # Finish run
    logger.finish()

    print("Logger test completed successfully!")


if __name__ == "__main__":
    main()