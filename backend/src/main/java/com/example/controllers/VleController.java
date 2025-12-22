package com.example.controllers;

import com.example.entities.learning.Vle;
import com.example.repositories.VleRepository;
import com.example.services.VleService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/vle")
public class VleController {

    @Autowired
    private VleService vleService;

    @GetMapping("/all")
    public List<Vle> getAllVle() {
        return vleService.getAllVles();

    }
}

