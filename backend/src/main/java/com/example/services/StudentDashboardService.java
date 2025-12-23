package com.example.services;

import com.example.dto.StudentDashboardDTO;
import com.example.dto.StudentDashboardDTO.*;
import com.example.entities.Etudiant;
import com.example.entities.learning.Assessment;
import com.example.entities.learning.StudentAssessment;
import com.example.entities.learning.StudentRegistration;
import com.example.repositories.AssessmentRepository;
import com.example.repositories.EtudiantRepository;
import com.example.repositories.StudentAssessmentRepository;
import com.example.repositories.StudentRegistrationRepository;
import com.example.repositories.UserRepository;
import com.example.entities.Utilisateur;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class StudentDashboardService {

    private final EtudiantRepository etudiantRepository;
    private final StudentRegistrationRepository registrationRepository;
    private final AssessmentRepository assessmentRepository;
    private final StudentAssessmentRepository studentAssessmentRepository;
    private final UserRepository userRepository; // Added

    public StudentDashboardService(
            EtudiantRepository etudiantRepository,
            StudentRegistrationRepository registrationRepository,
            AssessmentRepository assessmentRepository,
            StudentAssessmentRepository studentAssessmentRepository,
            UserRepository userRepository) { // Added
        this.etudiantRepository = etudiantRepository;
        this.registrationRepository = registrationRepository;
        this.assessmentRepository = assessmentRepository;
        this.studentAssessmentRepository = studentAssessmentRepository;
        this.userRepository = userRepository; // Added
    }

    public StudentDashboardDTO getDashboard(Integer studentId) {
        // Try to get student info from Etudiant table first
        StudentInfo studentInfo;
        try {
            Etudiant etudiant = etudiantRepository.findById(studentId)
                    .orElseThrow(() -> new RuntimeException("Student not found in Etudiant table"));
            studentInfo = new StudentInfo(etudiant.getNom(), etudiant.getPrenom());
        } catch (Exception e) {
            // Fallback: Check in Utilisateur table (for newly registered users)
            try {
                Utilisateur user = userRepository.findById(studentId)
                        .orElseThrow(() -> new RuntimeException("User not found"));
                studentInfo = new StudentInfo(user.getNom(), user.getPrenom());
            } catch (Exception ex) {
                // Final Fallback for demo/testing
                studentInfo = new StudentInfo("Étudiant", "Test (Mock)");
            }
        }

        // Calculate progression (mock for now - can be enhanced with real calculations)
        ProgressInfo progression = new ProgressInfo(
                calculateGlobalProgress(studentId),
                5 // Mock week change
        );

        // Get courses
        List<CourseProgress> courses = getCourseProgress(studentId);

        // Get urgent assignments
        List<UrgentAssignment> urgentAssignments = getUrgentAssignments(studentId);

        return new StudentDashboardDTO(studentInfo, progression, courses, urgentAssignments);
    }

    private int calculateGlobalProgress(Integer studentId) {
        // Mock calculation - can be replaced with actual logic
        List<StudentAssessment> assessments = studentAssessmentRepository.findByStudentId(studentId);
        if (assessments.isEmpty()) {
            return 0;
        }

        double avgScore = assessments.stream()
                .filter(sa -> sa.getScore() > 0)
                .mapToDouble(StudentAssessment::getScore)
                .average()
                .orElse(0.0);

        return (int) Math.min(100, avgScore);
    }

    private List<CourseProgress> getCourseProgress(Integer studentId) {
        List<StudentRegistration> registrations = registrationRepository.findByStudentId(studentId);

        return registrations.stream()
                .limit(5) // Top 5 courses
                .map(reg -> {
                    String courseModule = reg.getStudentRegistrationId().getCode_module();
                    int progress = calculateCourseProgress(studentId, courseModule);

                    return new CourseProgress(
                            courseModule,
                            progress,
                            getCourseIcon(courseModule));
                })
                .collect(Collectors.toList());
    }

    private int calculateCourseProgress(Integer studentId, String courseModule) {
        // Mock calculation - percentage of completed assessments
        List<StudentAssessment> courseAssessments = studentAssessmentRepository
                .findByStudentIdAndCodeModule(studentId, courseModule);

        if (courseAssessments.isEmpty()) {
            return 0;
        }

        long completed = courseAssessments.stream()
                .filter(sa -> sa.getScore() > 0)
                .count();

        return (int) ((completed * 100) / courseAssessments.size());
    }

    private String getCourseIcon(String courseModule) {
        // Map course modules to icons
        if (courseModule.contains("MAT") || courseModule.contains("MATH")) {
            return "math";
        } else if (courseModule.contains("INFO") || courseModule.contains("CS")) {
            return "computer";
        } else if (courseModule.contains("PHY")) {
            return "science";
        }
        return "book";
    }

    private List<UrgentAssignment> getUrgentAssignments(Integer studentId) {
        List<Assessment> allAssessments = assessmentRepository.findAll();
        List<StudentAssessment> studentAssessments = studentAssessmentRepository.findByStudentId(studentId);

        Set<Integer> completedAssessmentIds = studentAssessments.stream()
                .filter(sa -> sa.getScore() > 0)
                .map(sa -> sa.getStudentAssessmentId().getId_assessment())
                .collect(Collectors.toSet());

        LocalDate today = LocalDate.now();
        int currentDay = (int) ChronoUnit.DAYS.between(LocalDate.of(2024, 1, 1), today);

        return allAssessments.stream()
                .filter(a -> !completedAssessmentIds.contains(a.getId_assessment()))
                .filter(a -> a.getDate() > currentDay)
                .filter(a -> a.getDate() - currentDay <= 7) // Due within 7 days
                .sorted(Comparator.comparingInt(Assessment::getDate))
                .limit(5)
                .map(a -> new UrgentAssignment(
                        a.getAssessment_type() + " - " + a.getCode_module(),
                        a.getDate() - currentDay))
                .collect(Collectors.toList());
    }
}
