package com.example.entities.learning;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Entity
@Table(name = "studentVle")
public class StudentVle {
    @EmbeddedId
    private StudentVleId studentVleId;
    private int date;
    private int sum_click ;

    @ManyToOne
    @JoinColumn(name = "id_site")
    private Vle vle;

    @ManyToOne
    @JoinColumn(name = "id_student")
    private StudentInfo student;

    @ManyToOne
    @JoinColumns({
            @JoinColumn(name = "code_module", referencedColumnName = "codeModule", insertable=false, updatable=false),
            @JoinColumn(name = "code_presentation", referencedColumnName = "codePresentation", insertable=false, updatable=false)
    })
    private Courses course;

}
