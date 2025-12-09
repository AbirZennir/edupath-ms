package com.example.services;

import com.example.dto.ClassDetailDto;
import com.example.dto.ClassSummaryDto;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import static org.springframework.http.HttpStatus.NOT_FOUND;

@Service
public class ClassesService {

    private final List<ClassSummaryDto> summaries = List.of(
            new ClassSummaryDto("c1", "L3 Informatique – Algorithmique", "Semestre 5", "Licence", "Informatique", 45, 85, 78),
            new ClassSummaryDto("c2", "L2 Physique – Mécanique", "Semestre 4", "Licence", "Physique", 38, 78, 65),
            new ClassSummaryDto("c3", "M1 Data Science – Machine Learning", "Semestre 7", "Master", "Informatique", 32, 92, 88),
            new ClassSummaryDto("c4", "L3 Informatique – Base de données", "Semestre 5", "Licence", "Informatique", 42, 81, 72),
            new ClassSummaryDto("c5", "L2 Mathématiques – Algèbre linéaire", "Semestre 3", "Licence", "Mathématiques", 50, 75, 68),
            new ClassSummaryDto("c6", "M2 Intelligence Artificielle – Deep Learning", "Semestre 9", "Master", "Informatique", 28, 89, 82)
    );

    private final Map<String, ClassDetailDto> details = Map.of(
            "c1", new ClassDetailDto("c1", "L3 Informatique – Algorithmique", "Marie Petit", "Semestre 5", 45, 85, 42, 78, 82),
            "c2", new ClassDetailDto("c2", "L2 Physique – Mécanique", "Jean Martin", "Semestre 4", 38, 78, 34, 70, 75),
            "c3", new ClassDetailDto("c3", "M1 Data Science – Machine Learning", "Sophie Leroy", "Semestre 7", 32, 92, 30, 86, 88),
            "c4", new ClassDetailDto("c4", "L3 Informatique – Base de données", "Alice Dupont", "Semestre 5", 42, 81, 39, 73, 79),
            "c5", new ClassDetailDto("c5", "L2 Mathématiques – Algèbre linéaire", "Pierre Laurent", "Semestre 3", 50, 75, 46, 69, 71),
            "c6", new ClassDetailDto("c6", "M2 Intelligence Artificielle – Deep Learning", "Hugo Bernard", "Semestre 9", 28, 89, 26, 84, 87)
    );

    public List<ClassSummaryDto> list(String level, String category, String q) {
        return summaries.stream()
                .filter(c -> isBlank(level) || c.getLevel().equalsIgnoreCase(level))
                .filter(c -> isBlank(category) || c.getCategory().equalsIgnoreCase(category))
                .filter(c -> {
                    if (isBlank(q)) return true;
                    String query = q.toLowerCase(Locale.ROOT);
                    return c.getTitle().toLowerCase(Locale.ROOT).contains(query)
                            || c.getCategory().toLowerCase(Locale.ROOT).contains(query);
                })
                .collect(Collectors.toList());
    }

    public ClassDetailDto get(String id) {
        return Optional.ofNullable(details.get(id))
                .orElseThrow(() -> new ResponseStatusException(NOT_FOUND, "Classe introuvable"));
    }

    private boolean isBlank(String value) {
        return value == null || value.isBlank();
    }
}
