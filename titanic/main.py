from xvalid import CrossValidator
import data
from classifier import BasicSVM


def main():
    xv = CrossValidator(data.train)
    basicSVM = BasicSVM()
    report = xv.run(basicSVM)

    print("# Summary")
    print("\n```")
    print(report.summary)
    print("```\n")

    print("# Accuracy")
    print("\n```")
    print(report.accuracy)
    print("```\n")

if __name__ == "__main__":
    main()
