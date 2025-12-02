package com.example.entities.learning;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Entity
@Table(name = "courses")
public class Courses {
    @EmbeddedId
    private CourseId courseId;
    private int length;
    @OneToMany(mappedBy = "course")
    private List<Assessment> assessments;
    @OneToMany(mappedBy = "course")
    private List<Vle> vles;

}
