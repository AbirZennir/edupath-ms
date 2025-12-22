package com.example.services;

import com.example.dto.CourseItemDto;
import com.example.entities.learning.Courses;
import com.example.repositories.CoursesRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

@Service
public class CoursesService {

    @Autowired
    private CoursesRepository coursesRepository;

    private final List<CourseItemDto> seed = List.of(
            new CourseItemDto("Mathématiques Avancées", "Prof. Martin Dubois", "in_progress"),
            new CourseItemDto("Physique Quantique", "Prof. Sophie Laurent", "in_progress"),
            new CourseItemDto("Algorithmique", "Prof. Jean Dupont", "in_progress"),
            new CourseItemDto("Chimie Organique", "Prof. Marie Bernard", "done"),
            new CourseItemDto("Littérature Française", "Prof. Pierre Moreau", "in_progress")
    );

    public List<CourseItemDto> listCourses(String status, String search) {
        return seed.stream()
                .filter(c -> status == null || status.isBlank() || "all".equalsIgnoreCase(status) || c.getStatus().equalsIgnoreCase(status))
                .filter(c -> {
                    if (search == null || search.isBlank()) return true;
                    String q = search.toLowerCase(Locale.ROOT);
                    return c.getTitle().toLowerCase(Locale.ROOT).contains(q)
                            || c.getProfessor().toLowerCase(Locale.ROOT).contains(q);
                })
                .collect(Collectors.toList());
    }

    public List<Courses> getAllCourses() {
       return coursesRepository.findAll();
    }
}
