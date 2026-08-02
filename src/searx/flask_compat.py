import sys
import types
import contextvars

# ContextVar to store the current FastAPI request
fastapi_request_var = contextvars.ContextVar("fastapi_request", default=None)


class FlaskRequestProxy:
    def _get_current_object(self):
        return self

    @property
    def _req(self):
        req = fastapi_request_var.get()
        if req is None:

            class DummyRequest:
                remote_addr = "127.0.0.1"
                user_agent = "Mozilla/5.0"
                headers = {}
                form = {}
                args = {}
                cookies = {}
                path = "/"
                url = "http://localhost/"
                method = "GET"

            return DummyRequest()
        return req

    @property
    def remote_addr(self):
        client = getattr(self._req, "client", None)
        return client.host if client else "127.0.0.1"

    @property
    def user_agent(self):
        return self._req.headers.get("user-agent", "Mozilla/5.0")

    @property
    def headers(self):
        return self._req.headers

    @property
    def form(self):
        return getattr(self._req, "_form_data", {})

    @property
    def args(self):
        return getattr(self._req, "query_params", {})

    @property
    def cookies(self):
        return getattr(self._req, "cookies", {})

    @property
    def path(self):
        url = getattr(self._req, "url", None)
        return url.path if url else "/"

    @property
    def url(self):
        return str(getattr(self._req, "url", "http://localhost/"))

    @property
    def method(self):
        return getattr(self._req, "method", "GET")


# Mock module for flask
flask_mock = types.ModuleType("flask")
flask_mock.request = FlaskRequestProxy()
flask_mock.Request = FlaskRequestProxy


class DummyFlask:
    def __init__(self, *args, **kwargs):
        pass

    def after_request(self, func):
        return func

    def before_request(self, func):
        return func


flask_mock.Flask = DummyFlask
flask_mock.current_app = DummyFlask()
flask_mock.g = types.SimpleNamespace()


def dummy_copy_current_request_context(f):
    return f


flask_mock.copy_current_request_context = dummy_copy_current_request_context


class DummyResponse:
    def __init__(self, *args, **kwargs):
        self.headers = {}


flask_mock.Response = DummyResponse
flask_mock.make_response = lambda *args, **kwargs: DummyResponse()
flask_mock.abort = lambda *args, **kwargs: None
flask_mock.has_request_context = lambda: fastapi_request_var.get() is not None


class ctx_mock(types.ModuleType):
    has_request_context = lambda: fastapi_request_var.get() is not None


flask_mock.ctx = ctx_mock("flask.ctx")

sys.modules["flask"] = flask_mock
sys.modules["flask.ctx"] = flask_mock.ctx

# Create flask_babel mock
flask_babel_mock = types.ModuleType("flask_babel")
flask_babel_mock.gettext = lambda x, *args, **kwargs: x
flask_babel_mock.ngettext = lambda x, y, n, *args, **kwargs: x if n == 1 else y
flask_babel_mock.get_locale = lambda: None
flask_babel_mock.get_translations = lambda: None
flask_babel_mock.format_date = lambda *args, **kwargs: ""
flask_babel_mock.format_decimal = lambda x, *args, **kwargs: str(x)
flask_babel_mock.Babel = lambda *args, **kwargs: None

sys.modules["flask_babel"] = flask_babel_mock
