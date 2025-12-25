package com.example.services;

import com.example.entities.learning.*;
import com.example.repositories.*;
import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Service
public class CsvImportService {
    private static final Logger LOGGER = LoggerFactory.getLogger(CsvImportService.class);

    @Value("${csv.import.maxRows:10000}")
    private int maxRows;

    @Autowired
    private CoursesRepository coursesRepository;

    @Autowired
    private AssessmentRepository assessmentRepository;

    @Autowired
    private VleRepository vleRepository;

    @Autowired
    private StudentInfoRepository studentInfoRepository;

    @Autowired
    private StudentRegistrationRepository studentRegistrationRepository;

    @Autowired
    private StudentAssessmentRepository studentAssessmentRepository;

    @Autowired
    private StudentVleRepository studentVleRepository;

    private List<String[]> limitRows(List<String[]> rows) {
        if (maxRows > 0 && rows.size() > maxRows) {
            LOGGER.info("Limiting import to {} rows (total available: {})", maxRows, rows.size());
            return rows.subList(0, maxRows);
        }
        return rows;
    }

    @Transactional
    public void importCourses(String filePath) throws IOException, CsvException {
        LOGGER.info("Starting to import courses from: {}", filePath);

        try (CSVReader reader = new CSVReader(new FileReader(filePath))) {
            List<String[]> rows = reader.readAll();
            rows.remove(0); // Remove header
            rows = limitRows(rows);

            List<Courses> courses = new ArrayList<>();
            for (String[] row : rows) {
                CourseId courseId = new CourseId();
                courseId.setCode_module(row[0]);
                courseId.setCode_presentation(row[1]);

                Courses course = new Courses();
                course.setCourseId(courseId);
                course.setLength(Integer.parseInt(row[2]));
                courses.add(course);
            }

            coursesRepository.saveAll(courses);
            LOGGER.info("Successfully imported {} courses", courses.size());
        }
    }

    @Transactional
    public void importAssessments(String filePath) throws IOException, CsvException {
        LOGGER.info("Starting to import assessments from: {}", filePath);

        try (CSVReader reader = new CSVReader(new FileReader(filePath))) {
            List<String[]> rows = reader.readAll();
            rows.remove(0); // Remove header
            rows = limitRows(rows);

            List<Assessment> assessments = new ArrayList<>();
            for (String[] row : rows) {
                Assessment assessment = new Assessment();
                assessment.setCode_module(row[0]);
                assessment.setCode_presentation(row[1]);
                assessment.setId_assessment(Integer.parseInt(row[2]));
                assessment.setAssessment_type(row[3]);
                assessment.setDate(row[4].isEmpty() ? 0 : Integer.parseInt(row[4]));
                assessment.setWeight(row[5].isEmpty() ? 0 : Double.parseDouble(row[5]));
                assessments.add(assessment);
            }

            assessmentRepository.saveAll(assessments);
            LOGGER.info("Successfully imported {} assessments", assessments.size());
        }
    }

    @Transactional
    public void importVle(String filePath) throws IOException, CsvException {
        LOGGER.info("Starting to import VLE from: {}", filePath);

        try (CSVReader reader = new CSVReader(new FileReader(filePath))) {
            List<String[]> rows = reader.readAll();
            rows.remove(0); // Remove header
            rows = limitRows(rows);

            List<Vle> vles = new ArrayList<>();
            for (String[] row : rows) {
                Vle vle = new Vle();
                vle.setId_site(Integer.parseInt(row[0]));
                vle.setCode_module(row[1]);
                vle.setCode_presentation(row[2]);
                vle.setActivity_type(row[3]);
                vle.setWeek_from(row[4].isEmpty() ? 0 : Integer.parseInt(row[4]));
                vle.setWeek_to(row[5].isEmpty() ? 0 : Integer.parseInt(row[5]));
                vles.add(vle);
            }

            vleRepository.saveAll(vles);
            LOGGER.info("Successfully imported {} VLE records", vles.size());
        }
    }

    @Transactional
    public void importStudentInfo(String filePath) throws IOException, CsvException {
        LOGGER.info("Starting to import student info from: {}", filePath);

        try (CSVReader reader = new CSVReader(new FileReader(filePath))) {
            List<String[]> rows = reader.readAll();
            rows.remove(0); // Remove header
            rows = limitRows(rows);

            int batchSize = 1000;
            List<StudentInfo> batch = new ArrayList<>();

            for (String[] row : rows) {
                StudentInfo studentInfo = new StudentInfo();
                studentInfo.setCode_module(row[0]);
                studentInfo.setCode_presentation(row[1]);
                studentInfo.setId_student(Integer.parseInt(row[2]));
                studentInfo.setGender(row[3]);
                studentInfo.setRegion(row[4]);
                studentInfo.setHighest_education(row[5]);
                studentInfo.setImd_band(row[6]);
                studentInfo.setAge_band(row[7]);
                studentInfo.setNum_of_prev_attempts(row[8]);
                studentInfo.setStudied_credits(Integer.parseInt(row[9]));
                studentInfo.setDisability(row[10]);
                studentInfo.setFinal_result(row[11]);

                batch.add(studentInfo);

                if (batch.size() >= batchSize) {
                    studentInfoRepository.saveAll(batch);
                    batch.clear();
                    LOGGER.info("Saved batch of student info records");
                }
            }

            if (!batch.isEmpty()) {
                studentInfoRepository.saveAll(batch);
            }

            LOGGER.info("Successfully imported student info");
        }
    }

    @Transactional
    public void importStudentRegistration(String filePath) throws IOException, CsvException {
        LOGGER.info("Starting to import student registration from: {}", filePath);

        try (CSVReader reader = new CSVReader(new FileReader(filePath))) {
            List<String[]> rows = reader.readAll();
            rows.remove(0); // Remove header
            rows = limitRows(rows);

            int batchSize = 1000;
            List<StudentRegistration> batch = new ArrayList<>();
            int skipped = 0;

            for (String[] row : rows) {
                StudentRegistrationId id = new StudentRegistrationId();
                id.setCode_module(row[0]);
                id.setCode_presentation(row[1]);
                id.setId_student(Integer.parseInt(row[2]));

                // Check if student exists before inserting
                if (!studentInfoRepository.existsById(id.getId_student())) {
                    skipped++;
                    continue;
                }

                StudentRegistration registration = new StudentRegistration();
                registration.setStudentRegistrationId(id);
                registration.setDate_registration(Integer.parseInt(row[3]));
                registration.setDate_unregistration(row[4].isEmpty() ? 0 : Integer.parseInt(row[4]));

                batch.add(registration);

                if (batch.size() >= batchSize) {
                    studentRegistrationRepository.saveAll(batch);
                    batch.clear();
                    LOGGER.info("Saved batch of student registration records");
                }
            }

            if (!batch.isEmpty()) {
                studentRegistrationRepository.saveAll(batch);
            }

            if (skipped > 0) {
                LOGGER.warn("Skipped {} student registration records due to missing foreign keys", skipped);
            }
            LOGGER.info("Successfully imported {} student registration records", rows.size() - skipped);
        }
    }

    @Transactional
    public void importStudentAssessment(String filePath) throws IOException, CsvException {
        LOGGER.info("Starting to import student assessment from: {}", filePath);

        try (CSVReader reader = new CSVReader(new FileReader(filePath))) {
            List<String[]> rows = reader.readAll();
            rows.remove(0); // Remove header
            rows = limitRows(rows);

            int batchSize = 1000;
            List<StudentAssessment> batch = new ArrayList<>();
            int skipped = 0;

            for (String[] row : rows) {
                StudentAssessmentId id = new StudentAssessmentId();
                id.setId_assessment(Integer.parseInt(row[0]));
                id.setId_student(Integer.parseInt(row[1]));

                // Check if student and assessment exist before inserting
                if (!studentInfoRepository.existsById(id.getId_student())) {
                    skipped++;
                    continue;
                }
                if (!assessmentRepository.existsById(id.getId_assessment())) {
                    skipped++;
                    continue;
                }

                StudentAssessment assessment = new StudentAssessment();
                assessment.setStudentAssessmentId(id);
                assessment.setDate_submitted(row[2].isEmpty() ? 0 : Integer.parseInt(row[2]));
                assessment.setIs_banked(Integer.parseInt(row[3]));
                assessment.setScore(row[4].isEmpty() ? 0 : Float.parseFloat(row[4]));

                batch.add(assessment);

                if (batch.size() >= batchSize) {
                    studentAssessmentRepository.saveAll(batch);
                    batch.clear();
                    LOGGER.info("Saved batch of student assessment records");
                }
            }

            if (!batch.isEmpty()) {
                studentAssessmentRepository.saveAll(batch);
            }

            if (skipped > 0) {
                LOGGER.warn("Skipped {} student assessment records due to missing foreign keys", skipped);
            }
            LOGGER.info("Successfully imported {} student assessment records", rows.size() - skipped);
        }
    }

    @Transactional
    public void importStudentVle(String filePath) throws IOException, CsvException {
        LOGGER.info("Starting to import student VLE from: {}", filePath);

        try (CSVReader reader = new CSVReader(new FileReader(filePath))) {
            List<String[]> rows = reader.readAll();
            rows.remove(0); // Remove header
            rows = limitRows(rows);

            int batchSize = 1000;
            List<StudentVle> batch = new ArrayList<>();
            int count = 0;
            int skipped = 0;

            for (String[] row : rows) {
                StudentVleId id = new StudentVleId();
                id.setCode_module(row[0]);
                id.setCode_presentation(row[1]);
                id.setId_student(Integer.parseInt(row[2]));
                id.setId_site(Integer.parseInt(row[3]));

                // Check if student and VLE site exist before inserting
                if (!studentInfoRepository.existsById(id.getId_student())) {
                    skipped++;
                    continue;
                }
                if (!vleRepository.existsById(id.getId_site())) {
                    skipped++;
                    continue;
                }

                StudentVle studentVle = new StudentVle();
                studentVle.setStudentVleId(id);
                studentVle.setDate(Integer.parseInt(row[4]));
                studentVle.setSum_click(Integer.parseInt(row[5]));

                batch.add(studentVle);

                if (batch.size() >= batchSize) {
                    studentVleRepository.saveAll(batch);
                    count += batch.size();
                    batch.clear();
                    LOGGER.info("Saved {} student VLE records so far...", count);
                }
            }

            if (!batch.isEmpty()) {
                studentVleRepository.saveAll(batch);
                count += batch.size();
            }

            if (skipped > 0) {
                LOGGER.warn("Skipped {} student VLE records due to missing foreign keys", skipped);
            }
            LOGGER.info("Successfully imported {} student VLE records", count);
        }
    }

    public void importAllCsvFiles(String basePath) {
        try {
            // Import in order of dependencies
            LOGGER.info("=== Starting CSV Import Process ===");

            // 1. Import courses first (no dependencies)
            importCourses(basePath + "/courses.csv");

            // 2. Import assessments and VLE (depend on courses)
            importAssessments(basePath + "/assessments.csv");
            importVle(basePath + "/vle.csv");

            // 3. Import student info (no dependencies)
            importStudentInfo(basePath + "/studentInfo.csv");

            // 4. Import student registration (depends on courses and students)
            importStudentRegistration(basePath + "/studentRegistration.csv");

            // 5. Import student assessment (depends on assessments and students)
            importStudentAssessment(basePath + "/studentAssessment.csv");

            // 6. Import student VLE (depends on VLE and students) - largest file
            importStudentVle(basePath + "/studentVle.csv");

            LOGGER.info("=== CSV Import Process Completed Successfully ===");
        } catch (Exception e) {
            LOGGER.error("Error during CSV import process", e);
            throw new RuntimeException("Failed to import CSV files", e);
        }
    }
}

