package com.example.facialeditapp.page

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import com.example.facialeditapp.R
import java.util.*


class SaveImagePage : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.save_image_page)
        val myToolbar: Toolbar = findViewById(R.id.my_toolbar)
        setSupportActionBar(myToolbar)

        val outputString = intent.extras?.get(ChooseTypePage.OUTPUT) as String
        val outputByte = Base64.getDecoder().decode(outputString)
        val outputBmp = resizeBitmap(BitmapFactory.decodeByteArray(outputByte, 0, outputByte.size))

        val imageView = findViewById<ImageView>(R.id.outputImage)
        imageView.setImageBitmap(outputBmp)
    }

    private fun resizeBitmap(before: Bitmap): Bitmap {
        val height = ChooseTypePage.originalHeight
        val width  = ChooseTypePage.originalWidth

        return Bitmap.createScaledBitmap(before, width,  height,true)
    }
}