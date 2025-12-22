package com.example.repositories;

import com.example.entities.learning.StudentInfo;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface StudentInfoRepository extends JpaRepository<StudentInfo, Integer> {
    /**
     * Count successful students (pass or distinction)
     */
    @Query("SELECT COUNT(s) FROM StudentInfo s WHERE LOWER(s.final_result) IN ('pass', 'distinction')")
    long countSuccessfulStudents();

    /**
     * Count students by final result values
     */
    @Query("SELECT COUNT(s) FROM StudentInfo s WHERE s.final_result IN :results")
    long countByFinalResultIn(List<String> results);

    /**
     * Find students by code module and presentation
     */
    @Query("SELECT s FROM StudentInfo s WHERE s.code_module = :codeModule AND s.code_presentation = :codePresentation")
    List<StudentInfo> findByCode_moduleAndCode_presentation(@Param("codeModule") String codeModule, @Param("codePresentation") String codePresentation);
}
