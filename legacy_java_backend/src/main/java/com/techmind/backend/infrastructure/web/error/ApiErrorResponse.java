package com.techmind.backend.infrastructure.web.error;

import java.util.List;

public record ApiErrorResponse(
        String code,
        String message,
        List<FieldValidationError> fieldErrors
) {

    public ApiErrorResponse {
        fieldErrors = List.copyOf(fieldErrors);
    }
}
