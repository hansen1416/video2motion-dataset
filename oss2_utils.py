import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
from dotenv import load_dotenv

load_dotenv()


def get_bucket():

    auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())

    endpoint = "oss-ap-southeast-1.aliyuncs.com"

    # oss bucket, timeout 30s
    return oss2.Bucket(auth, endpoint, "pose-daten", connect_timeout=30)
