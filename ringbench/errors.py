"""User-visible validation errors."""

class WorkbenchError(Exception):
    def __init__(self, message, *, status=400, code="invalid_input", detail=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.detail = detail

    def payload(self):
        error = {"message": str(self), "code": self.code}
        if self.detail is not None:
            error["detail"] = self.detail
        return {"error": error}


class Missing(WorkbenchError):
    def __init__(self, message):
        super().__init__(message, status=404, code="missing")


class Changed(WorkbenchError):
    def __init__(self, revision):
        super().__init__(
            "工程已经在另一个页面中修改，请刷新工程后重试。",
            status=409,
            code="revision_changed",
            detail={"revision": revision},
        )
