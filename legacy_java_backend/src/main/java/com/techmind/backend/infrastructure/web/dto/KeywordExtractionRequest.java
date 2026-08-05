package com.techmind.backend.infrastructure.web.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public record KeywordExtractionRequest(
        @NotBlank(message = "El título es obligatorio")
        @Size(max = 200, message = "El título no puede superar los 200 caracteres")
        String title,

        @NotBlank(message = "El texto es obligatorio")
        @Size(max = 20_000, message = "El texto no puede superar los 20000 caracteres")
        String text
) {
}
