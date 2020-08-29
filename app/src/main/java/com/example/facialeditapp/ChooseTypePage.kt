package com.example.facialeditapp

import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar

class ChooseTypePage : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.choose_type_page)
        val myToolbar: Toolbar = findViewById(R.id.my_toolbar)
        setSupportActionBar(myToolbar)
    }

    public fun clickEditButton(view: View) {
        val intent = Intent(this@ChooseTypePage, SaveImagePage::class.java)
        startActivity(intent)
    }
}