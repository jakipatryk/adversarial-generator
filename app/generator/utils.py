import base64
import io


def img_to_base64(img):
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        img_string = base64.b64encode(output.getvalue())
        return img_string.decode("utf-8")
