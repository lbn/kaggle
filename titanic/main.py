from xvalid import CrossValidator
import data
from classifier import BasicSVM


def main():
    xv = CrossValidator(data.train)
    basicSVM = BasicSVM()
    report = xv.run(basicSVM)
    print(report)


if __name__ == "__main__":
    main()
