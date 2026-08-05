package com.techmind.backend.application.service;

import com.techmind.backend.domain.port.KeywordExtractor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class KeywordExtractionService {

    private final KeywordExtractor keywordExtractor;

    public KeywordExtractionService(KeywordExtractor keywordExtractor) {
        this.keywordExtractor = keywordExtractor;
    }

    public List<String> extract(String title, String text) {
        return keywordExtractor.extract(title, text);
    }
}
