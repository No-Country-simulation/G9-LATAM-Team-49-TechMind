package com.techmind.backend.domain.port;
import java.util.List;

public interface KeywordExtractor {
    List<String> extract(String  text);
}
