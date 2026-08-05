package com.techmind.backend.infrastructure.web;

import com.techmind.backend.application.service.KeywordExtractionService;
import com.techmind.backend.infrastructure.web.dto.KeywordExtractionRequest;
import com.techmind.backend.infrastructure.web.dto.KeywordExtractionResponse;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/keywords")
public class KeywordController {

    private final KeywordExtractionService keywordExtractionService;

    public KeywordController(KeywordExtractionService keywordExtractionService) {
        this.keywordExtractionService = keywordExtractionService;
    }

    @PostMapping
    public ResponseEntity<KeywordExtractionResponse> extract(
            @Valid @RequestBody KeywordExtractionRequest request
    ) {
        var keywords = keywordExtractionService.extract(request.title(), request.text());
        return ResponseEntity.ok(new KeywordExtractionResponse(keywords));
    }
}
