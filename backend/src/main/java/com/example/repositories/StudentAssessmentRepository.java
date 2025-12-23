package com.example.repositories;

import com.example.entities.learning.StudentAssessment;
import com.example.entities.learning.StudentAssessmentId;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface StudentAssessmentRepository extends JpaRepository<StudentAssessment, StudentAssessmentId> {

    @Query("select count(sa) from StudentAssessment sa " +
            "where sa.studentAssessmentId.id_student = :studentId " +
            "and (:codePres is null or sa.assessment.code_presentation = :codePres)")
    long countByStudent(@Param("studentId") int studentId, @Param("codePres") String codePresentation);

    @Query("select sa from StudentAssessment sa where sa.studentAssessmentId.id_student = :studentId")
    List<StudentAssessment> findByStudentId(@Param("studentId") int studentId);

    @Query("select sa from StudentAssessment sa where sa.studentAssessmentId.id_student = :studentId and sa.assessment.code_module = :codeModule")
    List<StudentAssessment> findByStudentIdAndCodeModule(@Param("studentId") int studentId,
            @Param("codeModule") String codeModule);
}
