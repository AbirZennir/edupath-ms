package com.example.repositories;

import com.example.entities.learning.StudentVle;
import com.example.entities.learning.StudentVleId;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface StudentVleRepository extends JpaRepository<StudentVle, StudentVleId> {

    @Query("select coalesce(sum(v.sum_click),0) from StudentVle v " +
            "where v.studentVleId.id_student = :studentId " +
            "and (:codePres is null or v.studentVleId.code_presentation = :codePres)")
    long sumClicksByStudent(@Param("studentId") int studentId, @Param("codePres") String codePresentation);
}
