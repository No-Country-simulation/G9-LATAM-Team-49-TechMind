package com.techmind.backend.infrastructure.extractor;

import com.techmind.backend.domain.port.KeywordExtractor;
import org.springframework.stereotype.Component;

import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

@Component
public class LocalKeywordExtractor implements KeywordExtractor {

    private static final int MAX_KEYWORDS = 5;
    private static final Pattern WORD_PATTERN = Pattern.compile("[\\p{L}\\p{N}+#]{3,}");
    private static final Set<String> STOP_WORDS = Set.of(
            "con", "del", "las", "los", "para", "por", "que", "una", "uno", "unos", "unas",
            "este", "esta", "estos", "estas", "como", "más", "pero", "sus", "sin", "sobre",
            "entre", "desde", "hasta", "donde", "cuando", "también", "muy", "son", "ser",
            "the", "and", "for", "from", "with", "this", "that", "into", "using"
    );

    @Override
    public List<String> extract(String title, String text) {
        var content = title + " " + title + " " + text;
        var matcher = WORD_PATTERN.matcher(content);
        Map<String, KeywordCandidate> candidates = new LinkedHashMap<>();
        var position = 0;

        while (matcher.find()) {
            var originalWord = matcher.group();
            var normalizedWord = originalWord.toLowerCase(Locale.ROOT);

            if (!STOP_WORDS.contains(normalizedWord)) {
                var candidate = candidates.get(normalizedWord);
                if (candidate == null) {
                    candidate = new KeywordCandidate(originalWord, position);
                    candidates.put(normalizedWord, candidate);
                }
                candidate.increment();
            }
            position++;
        }

        return candidates.values()
                .stream()
                .sorted(Comparator.comparingInt(KeywordCandidate::count).reversed()
                        .thenComparingInt(KeywordCandidate::firstPosition))
                .limit(MAX_KEYWORDS)
                .map(KeywordCandidate::displayValue)
                .toList();
    }

    private static final class KeywordCandidate {

        private final String displayValue;
        private final int firstPosition;
        private int count;

        private KeywordCandidate(String displayValue, int firstPosition) {
            this.displayValue = displayValue;
            this.firstPosition = firstPosition;
        }

        private void increment() {
            count++;
        }

        private String displayValue() {
            return displayValue;
        }

        private int firstPosition() {
            return firstPosition;
        }

        private int count() {
            return count;
        }
    }
}
