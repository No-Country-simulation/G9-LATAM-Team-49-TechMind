package com.techmind.backend.infrastructure.extractor;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class LocalKeywordExtractorTests {

    private final LocalKeywordExtractor extractor = new LocalKeywordExtractor();

    @Test
    void extractsAtMostFiveKeywordsOrderedByRelevance() {
        List<String> keywords = extractor.extract(
                "Introducción a Spring Boot",
                "Spring Boot permite crear APIs REST con Java. Spring simplifica la configuración."
        );

        assertEquals(5, keywords.size());
        assertEquals("Spring", keywords.getFirst());
        assertFalse(keywords.contains("con"));
    }
}
