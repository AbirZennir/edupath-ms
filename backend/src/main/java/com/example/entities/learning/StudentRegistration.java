package com.example.entities.learning;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Entity
@Table(name = "studentRegistration")
public class StudentRegistration {
    @EmbeddedId
    private StudentRegistrationId studentRegistrationId;

    private int date_registration;
    private int date_unregistration;

    @ManyToOne
    @JoinColumns({
            @JoinColumn(name = "code_module", referencedColumnName = "codeModule", insertable=false, updatable=false),
            @JoinColumn(name = "code_presentation", referencedColumnName = "codePresentation", insertable=false, updatable=false)
    })
    private Courses course;
}
