package com.example.repositories;

import com.example.entities.learning.StudentVle;
import com.example.entities.learning.StudentVleId;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface StudentVleRepository extends JpaRepository<StudentVle, StudentVleId> {

    @Query("select coalesce(sum(v.sum_click),0) from StudentVle v " +
            "where v.studentVleId.id_student = :studentId " +
            "and (:codePres is null or v.studentVleId.code_presentation = :codePres)")
    long sumClicksByStudent(@Param("studentId") int studentId, @Param("codePres") String codePresentation);

   
    @Query("SELECT COALESCE(SUM(sv.sum_click), 0) FROM StudentVle sv")
    Long sumAllClicks();

    
    @Query("SELECT sv.studentVleId.id_student, SUM(sv.sum_click) FROM StudentVle sv GROUP BY sv.studentVleId.id_student")
    List<Object[]> getTotalClicksPerStudent();

    
    @Query("SELECT sv.studentVleId.id_student, COUNT(sv), SUM(sv.sum_click), AVG(sv.sum_click) " +
            "FROM StudentVle sv GROUP BY sv.studentVleId.id_student")
    List<Object[]> getStudentEngagementMetrics();
}
