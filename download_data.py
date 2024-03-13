import zipfile
import requests
import io


def main():
    url = (
        "https://s3-us-west-2.amazonaws.com/ai2-website/data/OpenBookQA-V1-Sep2018.zip"
    )

    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()


if __name__ == "__main__":
    main()
