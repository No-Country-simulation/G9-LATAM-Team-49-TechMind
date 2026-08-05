package com.techmind.backend.infrastructure.web.dto;

import java.util.List;

public record KeywordExtractionResponse(List<String> keywords) {

    public KeywordExtractionResponse {
        keywords = List.copyOf(keywords);
    }
}
