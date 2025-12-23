package com.example.services;

import com.example.dto.CourseItemDto;
import com.example.entities.learning.Courses;
import com.example.entities.learning.StudentRegistration;
import com.example.repositories.CoursesRepository;
import com.example.repositories.StudentRegistrationRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

@Service
public class CoursesService {

    @Autowired
    private CoursesRepository coursesRepository;

    @Autowired
    private StudentRegistrationRepository registrationRepository;

    // Fallback seed data if no courses found
    private final List<CourseItemDto> seed = List.of(
            new CourseItemDto("Mathématiques Avancées", "Prof. Martin Dubois", "in_progress"),
            new CourseItemDto("Physique Quantique", "Prof. Sophie Laurent", "in_progress"),
            new CourseItemDto("Algorithmique", "Prof. Jean Dupont", "in_progress"),
            new CourseItemDto("Chimie Organique", "Prof. Marie Bernard", "done"),
            new CourseItemDto("Littérature Française", "Prof. Pierre Moreau", "in_progress"));

    public List<CourseItemDto> listCourses(String status, String search) {
        return seed.stream()
                .filter(c -> status == null || status.isBlank() || "all".equalsIgnoreCase(status)
                        || c.getStatus().equalsIgnoreCase(status))
                .filter(c -> {
                    if (search == null || search.isBlank())
                        return true;
                    String q = search.toLowerCase(Locale.ROOT);
                    return c.getTitle().toLowerCase(Locale.ROOT).contains(q)
                            || c.getProfessor().toLowerCase(Locale.ROOT).contains(q);
                })
                .collect(Collectors.toList());
    }

    public List<CourseItemDto> getStudentCourses(Integer studentId, String status, String search) {
        List<StudentRegistration> registrations = registrationRepository.findByStudentId(studentId);

        if (registrations.isEmpty()) {
            // Return seed data if no registrations found
            return listCourses(status, search);
        }

        return registrations.stream()
                .map(reg -> {
                    String courseModule = reg.getStudentRegistrationId().getCode_module();
                    String courseStatus = reg.getDate_unregistration() > 0 ? "done" : "in_progress";

                    return new CourseItemDto(
                            formatCourseName(courseModule),
                            "Prof. " + generateProfessorName(courseModule),
                            courseStatus);
                })
                .filter(c -> status == null || status.isBlank() || "all".equalsIgnoreCase(status)
                        || c.getStatus().equalsIgnoreCase(status))
                .filter(c -> {
                    if (search == null || search.isBlank())
                        return true;
                    String q = search.toLowerCase(Locale.ROOT);
                    return c.getTitle().toLowerCase(Locale.ROOT).contains(q)
                            || c.getProfessor().toLowerCase(Locale.ROOT).contains(q);
                })
                .collect(Collectors.toList());
    }

    private String formatCourseName(String courseModule) {
        if (courseModule == null)
            return "Cours";

        if (courseModule.contains("AAA"))
            return "Mathématiques Avancées";
        if (courseModule.contains("BBB"))
            return "Physique Quantique";
        if (courseModule.contains("CCC"))
            return "Algorithmique";
        if (courseModule.contains("DDD"))
            return "Chimie Organique";
        if (courseModule.contains("EEE"))
            return "Littérature Française";

        return courseModule;
    }

    private String generateProfessorName(String courseModule) {
        int hash = courseModule != null ? courseModule.hashCode() : 0;
        String[] firstNames = { "Martin", "Sophie", "Jean", "Marie", "Pierre", "Claire", "Luc", "Anne" };
        String[] lastNames = { "Dubois", "Laurent", "Dupont", "Bernard", "Moreau", "Petit", "Roux", "Blanc" };

        return firstNames[Math.abs(hash) % firstNames.length] + " " +
                lastNames[Math.abs(hash / 10) % lastNames.length];
    }

    public List<Courses> getAllCourses() {
        return coursesRepository.findAll();
    }
}
