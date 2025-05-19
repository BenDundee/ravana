from chromadb import Documents, EmbeddingFunction, Embeddings
from openai import OpenAI
from typing import Optional, Any
import random
import string


### Thank you: https://github.com/wpcfan/chroma/blob/6520d23420a4dd687cbd74f34fa6c6a5cc1425b7/chromadb/utils/embedding_functions.py#L53
class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-ada-002",
        organization_id: Optional[str] = None,
        api_base: Optional[str] = None,
        api_type: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_id: Optional[str] = None,
    ):
        """
        Initialize the OpenAIEmbeddingFunction.

        Args:
            api_key (str, optional): Your API key for the OpenAI API. If not
                provided, it will raise an error to provide an OpenAI API key.
            organization_id(str, optional): The OpenAI organization ID if applicable
            model_name (str, optional): The name of the model to use for text
                embeddings. Defaults to "text-embedding-ada-002".
            api_base (str, optional): The base path for the API. If not provided,
                it will use the base path for the OpenRouter API. This can be used to
                point to a different deployment, such as an Azure deployment.
            api_type (str, optional): The type of the API deployment. This can be
                used to specify a different deployment, such as 'azure'. If not
                provided, it will use the default OpenAI deployment.

        """
        try:
            import openai
        except ImportError:
            raise ValueError(
                "The openai python package is not installed. Please install it with `pip install openai`"
            )

        if api_key is not None:
            openai.api_key = api_key
        # If the api key is still not set, raise an error
        elif openai.api_key is None:
            raise ValueError(
                "Please provide an OpenAI API key. You can get one at https://platform.openai.com/account/api-keys"
            )

        if api_base is not None:
            openai.api_base = api_base

        if api_type is not None:
            openai.api_type = api_type

        if api_version is not None:
            openai.api_version = api_version

        if organization_id is not None:
            openai.organization = organization_id

        self.deployment_id = deployment_id
        self._client = OpenAI(api_key=api_key).embeddings
        self._model_name = model_name

    def __call__(self, texts: Documents) -> list[Any] | list[list[float]]:
        # replace newlines, which can negatively affect performance.
        texts = [t.replace("\n", " ") for t in texts]

        # Call the OpenAI Embedding API
        if self.deployment_id is not None:
            print(f"Using deployment_id {self.deployment_id}")
            embeddings = []
            for text in texts:
                result = self._client.create(
                    model=self._model_name
                    , input=text
                    , encoding_format="float"
                ).data
                # TypeError: list indices must be integers or slices, not str
                json = result[0].enbedding
                embeddings.append(json)
            return embeddings
        else:
            embeddings = self._client.create(
                model=self._model_name
                , input=texts
                , encoding_format="float"
            ).data
            # Sort resulting embeddings by index
            sorted_embeddings = sorted(embeddings, key=lambda e: e.index)  # type: ignore
            return [result.embedding for result in sorted_embeddings]


def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string
