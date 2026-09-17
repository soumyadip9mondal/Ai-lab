# ============================================================
# LOGISTIC REGRESSION VISUALIZER
# GOOGLE COLAB
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

from IPython.display import display, clear_output

from sklearn.linear_model import LogisticRegression


# ============================================================
# 2. LOGISTIC REGRESSION VISUALIZER
# ============================================================

class LogisticRegressionVisualizer:

    def __init__(self):

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        self.X = []
        self.y = []

        self.model = None


        # ----------------------------------------------------
        # Widgets
        # ----------------------------------------------------

        self.test_slider = widgets.FloatSlider(

            value=5.0,

            min=0,

            max=10,

            step=0.1,

            description="Test X:",

            continuous_update=True,

            style={
                "description_width": "70px"
            },

            layout=widgets.Layout(
                width="350px"
            )
        )


        self.train_button = widgets.Button(

            description="Train Model",

            button_style="success",

            icon="check"
        )


        self.example_button = widgets.Button(

            description="Load Example",

            button_style="info"
        )


        self.reset_button = widgets.Button(

            description="Reset",

            button_style="warning"
        )


        self.add_class0_button = widgets.Button(

            description="Add Class 0"
        )


        self.add_class1_button = widgets.Button(

            description="Add Class 1"
        )


        self.x_input = widgets.FloatText(

            value=3.0,

            description="X value:",

            style={
                "description_width": "70px"
            }
        )


        # ----------------------------------------------------
        # Output Areas
        # ----------------------------------------------------

        self.plot_output = widgets.Output()

        self.info_output = widgets.Output()


        # ----------------------------------------------------
        # Button Actions
        # ----------------------------------------------------

        self.train_button.on_click(

            self.train_model
        )


        self.example_button.on_click(

            self.load_example
        )


        self.reset_button.on_click(

            self.reset
        )


        self.add_class0_button.on_click(

            self.add_class0
        )


        self.add_class1_button.on_click(

            self.add_class1
        )


        self.test_slider.observe(

            self.update_test_point,

            names="value"
        )


        # ----------------------------------------------------
        # Display Interface
        # ----------------------------------------------------

        self.show_interface()

        self.draw()


    # ========================================================
    # INTERFACE
    # ========================================================

    def show_interface(self):

        title = widgets.HTML(

            "<h2>Logistic Regression Visualizer</h2>"
        )


        instructions = widgets.HTML(

            """
            <b>How to use:</b><br>

            1. Load the example dataset, OR add your own points.<br>
            2. Class 0 = negative class.<br>
            3. Class 1 = positive class.<br>
            4. Click <b>Train Model</b>.<br>
            5. Move the Test X slider and watch the probability change.
            """
        )


        add_points = widgets.HBox([

            self.x_input,

            self.add_class0_button,

            self.add_class1_button
        ])


        buttons = widgets.HBox([

            self.example_button,

            self.train_button,

            self.reset_button
        ])


        display(

            title,

            instructions,

            widgets.HTML("<hr>"),

            add_points,

            buttons,

            self.test_slider,

            self.info_output,

            self.plot_output
        )


    # ========================================================
    # ADD CLASS 0
    # ========================================================

    def add_class0(self, button):

        x = self.x_input.value

        self.X.append([x])

        self.y.append(0)

        self.model = None

        self.draw()


    # ========================================================
    # ADD CLASS 1
    # ========================================================

    def add_class1(self, button):

        x = self.x_input.value

        self.X.append([x])

        self.y.append(1)

        self.model = None

        self.draw()


    # ========================================================
    # LOAD EXAMPLE
    # ========================================================

    def load_example(self, button=None):

        self.X = [

            [1.0],
            [1.5],
            [2.0],
            [2.5],
            [3.0],
            [3.2],
            [3.5],
            [3.8],
            [4.0],
            [4.2],
            [4.5],
            [5.0],
            [5.5],
            [6.0],
            [6.5],
            [7.0],
            [7.5],
            [8.0],
            [8.5]
        ]


        self.y = [

            0, 0, 0, 0, 0,

            0, 0, 0, 0,

            1, 1, 1, 1, 1,

            1, 1, 1, 1, 1
        ]


        self.model = None

        self.draw()


    # ========================================================
    # TRAIN MODEL
    # ========================================================

    def train_model(self, button=None):

        # ----------------------------------------------------
        # Check number of points
        # ----------------------------------------------------

        if len(self.X) < 4:

            with self.info_output:

                clear_output(wait=True)

                print(
                    "Please add at least 4 data points."
                )

            return


        # ----------------------------------------------------
        # Check both classes
        # ----------------------------------------------------

        if len(set(self.y)) < 2:

            with self.info_output:

                clear_output(wait=True)

                print(
                    "You need both Class 0 and Class 1."
                )

            return


        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        self.model = LogisticRegression()


        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        self.model.fit(

            np.array(self.X),

            np.array(self.y)
        )


        # ----------------------------------------------------
        # Draw result
        # ----------------------------------------------------

        self.draw()


    # ========================================================
    # DRAW EVERYTHING
    # ========================================================

    def draw(self):

        with self.plot_output:

            clear_output(wait=True)


            fig, ax = plt.subplots(

                figsize=(11, 6)
            )


            # ------------------------------------------------
            # Axis
            # ------------------------------------------------

            ax.set_xlim(

                0,

                10
            )


            ax.set_ylim(

                -0.1,

                1.1
            )


            ax.set_xlabel(

                "Feature X",

                fontsize=12
            )


            ax.set_ylabel(

                "Probability / Class",

                fontsize=12
            )


            ax.set_title(

                "Logistic Regression",

                fontsize=16,

                fontweight="bold"
            )


            ax.grid(

                alpha=0.3
            )


            # ------------------------------------------------
            # Plot training data
            # ------------------------------------------------

            if len(self.X) > 0:

                X_array = np.array(
                    self.X
                ).flatten()


                y_array = np.array(
                    self.y
                )


                class0 = X_array[
                    y_array == 0
                ]


                class1 = X_array[
                    y_array == 1
                ]


                # Class 0

                if len(class0) > 0:

                    ax.scatter(

                        class0,

                        np.zeros(
                            len(class0)
                        ),

                        s=100,

                        marker="o",

                        label="Class 0"
                    )


                # Class 1

                if len(class1) > 0:

                    ax.scatter(

                        class1,

                        np.ones(
                            len(class1)
                        ),

                        s=100,

                        marker="o",

                        label="Class 1"
                    )


            # ------------------------------------------------
            # Logistic Regression Curve
            # ------------------------------------------------

            if self.model is not None:

                x_values = np.linspace(

                    0,

                    10,

                    500
                ).reshape(

                    -1,

                    1
                )


                probabilities = (

                    self.model
                    .predict_proba(
                        x_values
                    )[:, 1]
                )


                ax.plot(

                    x_values,

                    probabilities,

                    linewidth=3,

                    label="P(Class 1)"
                )


                # --------------------------------------------
                # Threshold
                # --------------------------------------------

                ax.axhline(

                    0.5,

                    linestyle="--",

                    linewidth=1.5,

                    label="Threshold = 0.5"
                )


                # --------------------------------------------
                # Decision Boundary
                # --------------------------------------------

                coefficient = (

                    self.model.coef_[0][0]
                )


                intercept = (

                    self.model.intercept_[0]
                )


                if coefficient != 0:

                    boundary = (

                        -intercept /
                        coefficient
                    )


                    if 0 <= boundary <= 10:

                        ax.axvline(

                            boundary,

                            linestyle="--",

                            linewidth=2,

                            label=(
                                f"Decision Boundary = "
                                f"{boundary:.2f}"
                            )
                        )


                # --------------------------------------------
                # Test Point
                # --------------------------------------------

                test_x = self.test_slider.value


                probability = (

                    self.model
                    .predict_proba(
                        [[test_x]]
                    )[0][1]
                )


                prediction = (

                    self.model
                    .predict(
                        [[test_x]]
                    )[0]
                )


                ax.scatter(

                    test_x,

                    probability,

                    s=250,

                    marker="*",

                    label="Test Point"
                )


                # Vertical line

                ax.axvline(

                    test_x,

                    linestyle=":",

                    alpha=0.5
                )


            # ------------------------------------------------
            # Legend
            # ------------------------------------------------

            ax.legend(

                loc="best"
            )


            plt.show()


        # Update information

        self.update_information()


    # ========================================================
    # UPDATE INFORMATION
    # ========================================================

    def update_information(self, change=None):

        with self.info_output:

            clear_output(
                wait=True
            )


            print(

                f"Training samples: "
                f"{len(self.X)}"
            )


            # ------------------------------------------------
            # Model not trained
            # ------------------------------------------------

            if self.model is None:

                print(
                    "\nModel not trained."
                )

                return


            # ------------------------------------------------
            # Test point
            # ------------------------------------------------

            test_x = self.test_slider.value


            probability = (

                self.model
                .predict_proba(
                    [[test_x]]
                )[0][1]
            )


            prediction = (

                self.model
                .predict(
                    [[test_x]]
                )[0]
            )


            # ------------------------------------------------
            # Model parameters
            # ------------------------------------------------

            coefficient = (

                self.model.coef_[0][0]
            )


            intercept = (

                self.model.intercept_[0]
            )


            # ------------------------------------------------
            # Decision boundary
            # ------------------------------------------------

            if coefficient != 0:

                boundary = (

                    -intercept /
                    coefficient
                )

            else:

                boundary = float("inf")


            # ------------------------------------------------
            # Display model information
            # ------------------------------------------------

            print(
                "\n========== MODEL =========="
            )


            print(

                f"Coefficient (β₁): "
                f"{coefficient:.4f}"
            )


            print(

                f"Intercept (β₀): "
                f"{intercept:.4f}"
            )


            print(
                "\nz = β₀ + β₁X"
            )


            print(

                f"z = {intercept:.4f} "
                f"+ ({coefficient:.4f})X"
            )


            # ------------------------------------------------
            # Test point information
            # ------------------------------------------------

            print(
                "\n========== TEST POINT =========="
            )


            print(

                f"X = {test_x:.1f}"
            )


            print(

                f"P(Class 1) = "
                f"{probability:.4f}"
            )


            print(

                f"P(Class 0) = "
                f"{1 - probability:.4f}"
            )


            print(

                f"\nDecision Boundary = "
                f"{boundary:.4f}"
            )


            print(

                f"\nPrediction = "
                f"CLASS {prediction}"
            )


            # ------------------------------------------------
            # Classification decision
            # ------------------------------------------------

            if probability >= 0.5:

                print(
                    "\nP(Class 1) ≥ 0.5"
                )

                print(
                    "Therefore → CLASS 1"
                )

            else:

                print(
                    "\nP(Class 1) < 0.5"
                )

                print(
                    "Therefore → CLASS 0"
                )


    # ========================================================
    # UPDATE TEST POINT
    # ========================================================

    def update_test_point(self, change):

        if self.model is not None:

            self.draw()


    # ========================================================
    # RESET
    # ========================================================

    def reset(self, button=None):

        self.X = []

        self.y = []

        self.model = None

        self.draw()


# ============================================================
# START VISUALIZER
# ============================================================

visualizer = LogisticRegressionVisualizer()
