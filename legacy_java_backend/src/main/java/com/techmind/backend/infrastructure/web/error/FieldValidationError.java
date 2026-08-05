package com.techmind.backend.infrastructure.web.error;

public record FieldValidationError(String field, String message) {
}
