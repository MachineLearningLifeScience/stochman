from trainer_ae import train_ae
from trainer_lae import train_lae, test_lae


if __name__ == "__main__":

    dataset = "swissrole"

    # train or load auto encoder
    print("==> train ae")
    train_ae(dataset)

    print("==> test ae")
    test_ae(dataset)

    # train or load laplace approx
    print("==> train laplace ae")
    train_lae(dataset)

    # evaluate laplace auto encoder
    print("==> evaluate lae")
    test_lae(dataset)