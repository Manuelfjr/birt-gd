[![https://badgen.net/pypi/v/birt-gd](https://badgen.net/pypi/v/birt-gd)](https://pypi.org/project/birt-gd/#history)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/birt-gd?style=flat-square&color=darkgreen)](https://pypi.org/project/birt-gd/)
[![PyPI - Downloads](https://img.shields.io/pypi/dw/birt-gd?style=flat-square&color=darkgreen)](https://pypi.org/project/birt-gd/)
[![PyPI - Downloads](https://img.shields.io/pypi/dd/birt-gd?style=flat-square&color=darkgreen)](https://pypi.org/project/birt-gd/)
[![license: GPLv3](https://img.shields.io/badge/license-GPLv3-red.svg?&logo=license&color=blue)](https://github.com/Manuelfjr/birt-gd/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-birtgd-blue?&logo)](https://github.com/Manuelfjr/birt-gd)
[![Author](https://img.shields.io/badge/author-manuelfjr-blue?&logo=github)](https://github.com/Manuelfjr)
[![Author2](https://img.shields.io/badge/author-tmfilho-blue?&logo=github)](https://github.com/tmfilho)
[![https://badgen.net/github/open-issues/manuelfjr/birt-gd](https://badgen.net/github/open-issues/manuelfjr/birt-gd)](https://github.com/Manuelfjr/birt-gd/issues?q=is%3Aopen+is%3Aissue)
[![https://badgen.net/github/closed-issues/manuelfjr/birt-gd](https://badgen.net/github/closed-issues/manuelfjr/birt-gd)](https://github.com/Manuelfjr/birt-gd/issues?q=is%3Aissue+is%3Aclosed)

<!-- PyPi Status
![PyPI - Status](https://img.shields.io/pypi/status/pandas)
-->

<!--
![Py Coverage](https://s3.amazonaws.com/assets.coveralls.io/badges/coveralls_94.png)
-->

<!-- PyPi Downloads
[![PyPi - Downloads](https://pypip.in/d/pandas/badge.png?&color=blue&logo=python)](https://pypi.org/project/pandas/#files)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/scikit-learn?style=flat)](https://pypi.org/project/pandas/#files)
-->

<!-- Latest PyPI version
[![Latest PyPI version](https://img.shields.io/pypi/v/pandas?logo=pypi)](https://pypi.python.org/pypi/pandas)
-->

<!-- Release
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/pandas-dev/pandas)](https://GitHub.com/pandas-dev/pandas/releases/)

[![GitHub release](https://img.shields.io/github/release/Manuelfjr/birt-sgd.svg)](https://GitHub.com/Manuelfjr/birt-sgd/releases/)
-->

<!-- Static download of pepy
[![Downloads](https://static.pepy.tech/personalized-badge/pandas?period=total&units=international_system&left_color=grey&right_color=red&left_text=downloads)](https://pepy.tech/project/pandas)
-->

<!-- Github downloads
[![Github All Releases](https://img.shields.io/github/downloads/pandas-dev/pandas/total.svg?&logo=github&color=blue)]()
-->

<!-- Lines of code
![Lines of code](https://img.shields.io/tokei/lines/github/Manuelfjr/birt-sgd)
-->

<!-- Code size
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/pandas-dev/pandas)
-->

<!-- Github contributors
![GitHub contributors](https://img.shields.io/github/contributors/pandas-dev/pandas)
-->

<!--
[![Downloads](https://pepy.tech/badge/pandas)](https://pepy.tech/project/pandas)    
-->

# [birt-gd](https://pypi.org/project/birt-gd/)

**BIRTGD** is an implementation of &beta;<sup>3</sup>-irt using gradient descent.

The model expects to receive two sets of data, *X* being a list or array containing tuples of indices, where the first index references the instance *j* and the second index of the tuple references the model *i*, thus, *Y* will be a list or array where each input will be p<sub>ij</sub> ~ &Beta;(&alpha;<sub>ij</sub>, &beta;<sub>ij</sub>), the probability of the *i* model correctly classifying the *j* model. Being, 

p<sub>ij</sub> ~ &Beta;(&alpha;<sub>ij</sub>, &beta;<sub>ij</sub>)

&alpha;<sub>ij</sub> = F<sub>&alpha;</sub>(&theta;<sub>i</sub>, &delta;<sub>j</sub>, a<sub>j</sub>) = (&theta;<sub>i</sub>/&delta;<sub>j</sub>)<sup>a<sub>j</sub></sup>

&beta;<sub>ij</sub> = F<sub>&beta;</sub>(&theta;<sub>i</sub>, &delta;<sub>j</sub>, a<sub>j</sub>) = ( (1 - &theta;<sub>i</sub>)/(1 - &delta;<sub>j</sub>) )<sup>a<sub>j</sub></sup>

&theta;<sub>i</sub> ~ &Beta;(1,1), &delta;<sub>j</sub> ~ &Beta;(1,1), a<sub>j</sub> ~ N(1, &sigma;<sup>2</sup><sub>0</sub>)

where,

E[p<sub>ij</sub> | &theta;<sub>i</sub>, &delta;<sub>j</sub>, a<sub>j</sub>] = (&alpha;<sub>ij</sub>)/( &alpha;<sub>ij</sub> + &beta;<sub>ij</sub>) = 1/(1 + ( (&delta;<sub>j</sub>)/(1 - &delta;<sub>j</sub>) )<sup>a<sub>j</sub></sup> &#xd7; ( (&theta;<sub>i</sub>)/(1 - &theta;<sub>i</sub>) )<sup> - a<sub>j</sub></sup> )

# Installation
## Dependencies 
birt-gd requires:
- Python (>=3.6.0)
- numpy (>=1.19.5)
- tqdm (>=4.59.0)
- tensorflow (>=2.4.1)
- pandas (>=1.2.3)
- seaborn (>=0.11.0)
- matplotlib (>=3.3.2)
- scikit-learn (>=0.23.2)

## User installation

```bash
pip install birt-gd
```

## Source code 
You can check the code with 
```bash
git clone https://github.com/Manuelfjr/birt-gd
```

# Usage
Import the **BIRTGD's class**

```py
>>> from birt import BIRTGD
```

```py
>>> data = pd.DataFrame({'a': [0.99,0.89,0.87], 'b': [0.32,0.25,0.45]})
```

```py
>>> bgd = BIRTGD(n_models=2, n_instances=3, random_seed=1)
>>> bgd.fit(data)
100%|██████████| 5000/5000 [00:22<00:00, 219.50it/s]
<birt.BIRTGD at 0x7f6131326c10>
```

```py 
>>> bgd.abilities
array([0.90438306, 0.27729774], dtype=float32)
```

```py
>>> bgd.difficulties
array([0.3760659 , 0.5364428 , 0.34256178], dtype=float32)
```

```py
>>> bgd.discriminations
array([1.6690203 , 0.9951777 , 0.65577406], dtype=float32)
```

# Summary data

How to use the summary feature:

* **Generate data**
```py
import numpy as np
import pandas as pd
from birt import BIRTGD
import matplotlib.pyplot as plt

m, n = 5, 20
np.random.seed(1)
abilities = [np.random.beta(1,i) for i in ([0.1, 10] + [1]*(m-2))]
difficulties = [np.random.beta(1,i) for i in [10, 5] + [1]*(n-2)]
discrimination = list(np.random.normal(1,1, size=n))
pij = pd.DataFrame(columns=range(m), index=range(n))

i,j = 0,0
for theta in abilities:
  for delta, a in zip(difficulties, discrimination):
    alphaij = (theta/delta)**(a)
    betaij = ((1-theta)/(1 - delta))**(a)
    pij.loc[j,i] = np.random.beta(alphaij, betaij, size=1)[0]
    j+=1
  j = 0
  i+=1
```

* **Fitting the model**
```py
birt = BIRTGD(n_models=pij.shape[1],
             n_instances=pij.shape[0],
             learning_rate=1,
             epochs=5000,
             n_inits=1000)
birt.fit(pij)
```


* **Score (Pseudo - R<sup>2</sup>)**
```py
birt.score
```
```py
0.9038145665424927
```


* **Summary**
```py
birt.summary()
```
```py

        ESTIMATES
        -----
                        | Min      1Qt      Median   3Qt      Max      Std.Dev
        Ability         | 0.00010  0.22148  0.63389  0.73353  0.92040  0.33960
        Difficulty      | 0.01745  0.28047  0.63058  0.84190  0.98624  0.31635
        Discrimination  | 0.31464  1.28330  1.61493  2.22936  4.44645  1.02678
        pij             | 0.00000  0.02219  0.35941  0.86255  0.99993  0.40210
        -----
        Pseudo-R2       | 0.90381
        

```

# Using Scatterplot Feature

```py
birt.plot(xaxis='discrimination',
          yaxis='difficulty',
          ann=True,
          kwargs={'color': 'red'})
plt.show()
```

<img alt = "assets/dis_diff_ex.png" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/dis_diff_ex.png">

```py
birt.plot(xaxis='difficulty',
          yaxis='average_item',
          ann=True,
          kwargs={'color': 'blue'})
plt.show()
```

<img alt = "assets/diff_av_ex2.png" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/diff_av_ex2.png">


```py
birt.plot(xaxis='ability',
          yaxis='average_response',
          ann=False)
plt.show()
```

<img alt = "assets/ab_av_ex3.png" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/ab_av_ex3.png">

# Using Boxplot Feature

```py
birt.boxplot(y='abilities',
             kwargs={'linewidth': 4})
```

<img alt = "assets/ab_av_ex4.png" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/ex4.png">


```py
birt.boxplot(x='difficulties')
```
<img alt = "assets/ab_av_ex5.png" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/ex5.png">

```py
birt.boxplot(y='discriminations')
```

<img alt = "assets/ab_av_ex6.png" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/ex6.png">


# Help and Support
## Communication

- E-mail: [ferreira.jr.ufpb@gmail.com](mailto:ferreira.jr.ufpb@gmail.com)
- Site: [https://manuelfjr.github.io/](https://manuelfjr.github.io/)

# License
[GNU General Public License v3.0](https://github.com/Manuelfjr/birt-sgd/blob/main/LICENSE)

                    GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The GNU General Public License is a free, copyleft license for
software and other kinds of works.

  The licenses for most software and other practical works are designed
to take away your freedom to share and change the works.  By contrast,
the GNU General Public License is intended to guarantee your freedom to
share and change all versions of a program--to make sure it remains free
software for all its users.  We, the Free Software Foundation, use the
GNU General Public License for most of our software; it applies also to
any other work released this way by its authors.  You can apply it to
your programs, too.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
them if you wish), that you receive source code or can get it if you
want it, that you can change the software or use pieces of it in new
free programs, and that you know you can do these things.

  To protect your rights, we need to prevent others from denying you
these rights or asking you to surrender the rights.  Therefore, you have
certain responsibilities if you distribute copies of the software, or if
you modify it: responsibilities to respect the freedom of others.

  For example, if you distribute copies of such a program, whether
gratis or for a fee, you must pass on to the recipients the same
freedoms that you received.  You must make sure that they, too, receive
or can get the source code.  And you must show them these terms so they
know their rights.

  Developers that use the GNU GPL protect your rights with two steps:
(1) assert copyright on the software, and (2) offer you this License
giving you legal permission to copy, distribute and/or modify it.

  For the developers' and authors' protection, the GPL clearly explains
that there is no warranty for this free software.  For both users' and
authors' sake, the GPL requires that modified versions be marked as
changed, so that their problems will not be attributed erroneously to
authors of previous versions.

  Some devices are designed to deny users access to install or run
modified versions of the software inside them, although the manufacturer
can do so.  This is fundamentally incompatible with the aim of
protecting users' freedom to change the software.  The systematic
pattern of such abuse occurs in the area of products for individuals to
use, which is precisely where it is most unacceptable.  Therefore, we
have designed this version of the GPL to prohibit the practice for those
products.  If such problems arise substantially in other domains, we
stand ready to extend this provision to those domains in future versions
of the GPL, as needed to protect the freedom of users.

  Finally, every program is threatened constantly by software patents.
States should not allow patents to restrict development and use of
software on general-purpose computers, but in those that do, we wish to
avoid the special danger that patents applied to a free program could
make it effectively proprietary.  To prevent this, the GPL assures that
patents cannot be used to render the program non-free.

  The precise terms and conditions for copying, distribution and
modification follow.

                       TERMS AND CONDITIONS

  0. Definitions.

  "This License" refers to version 3 of the GNU General Public License.

  "Copyright" also means copyright-like laws that apply to other kinds of
works, such as semiconductor masks.

  "The Program" refers to any copyrightable work licensed under this
License.  Each licensee is addressed as "you".  "Licensees" and
"recipients" may be individuals or organizations.

  To "modify" a work means to copy from or adapt all or part of the work
in a fashion requiring copyright permission, other than the making of an
exact copy.  The resulting work is called a "modified version" of the
earlier work or a work "based on" the earlier work.

  A "covered work" means either the unmodified Program or a work based
on the Program.

  To "propagate" a work means to do anything with it that, without
permission, would make you directly or secondarily liable for
infringement under applicable copyright law, except executing it on a
computer or modifying a private copy.  Propagation includes copying,
distribution (with or without modification), making available to the
public, and in some countries other activities as well.

  To "convey" a work means any kind of propagation that enables other
parties to make or receive copies.  Mere interaction with a user through
a computer network, with no transfer of a copy, is not conveying.

  An interactive user interface displays "Appropriate Legal Notices"
to the extent that it includes a convenient and prominently visible
feature that (1) displays an appropriate copyright notice, and (2)
tells the user that there is no warranty for the work (except to the
extent that warranties are provided), that licensees may convey the
work under this License, and how to view a copy of this License.  If
the interface presents a list of user commands or options, such as a
menu, a prominent item in the list meets this criterion.

  1. Source Code.

  The "source code" for a work means the preferred form of the work
for making modifications to it.  "Object code" means any non-source
form of a work.

  A "Standard Interface" means an interface that either is an official
standard defined by a recognized standards body, or, in the case of
interfaces specified for a particular programming language, one that
is widely used among developers working in that language.

  The "System Libraries" of an executable work include anything, other
than the work as a whole, that (a) is included in the normal form of
packaging a Major Component, but which is not part of that Major
Component, and (b) serves only to enable use of the work with that
Major Component, or to implement a Standard Interface for which an
implementation is available to the public in source code form.  A
"Major Component", in this context, means a major essential component
(kernel, window system, and so on) of the specific operating system
(if any) on which the executable work runs, or a compiler used to
produce the work, or an object code interpreter used to run it.

  The "Corresponding Source" for a work in object code form means all
the source code needed to generate, install, and (for an executable
work) run the object code and to modify the work, including scripts to
control those activities.  However, it does not include the work's
System Libraries, or general-purpose tools or generally available free
programs which are used unmodified in performing those activities but
which are not part of the work.  For example, Corresponding Source
includes interface definition files associated with source files for
the work, and the source code for shared libraries and dynamically
linked subprograms that the work is specifically designed to require,
such as by intimate data communication or control flow between those
subprograms and other parts of the work.

  The Corresponding Source need not include anything that users
can regenerate automatically from other parts of the Corresponding
Source.

  The Corresponding Source for a work in source code form is that
same work.

  2. Basic Permissions.

  All rights granted under this License are granted for the term of
copyright on the Program, and are irrevocable provided the stated
conditions are met.  This License explicitly affirms your unlimited
permission to run the unmodified Program.  The output from running a
covered work is covered by this License only if the output, given its
content, constitutes a covered work.  This License acknowledges your
rights of fair use or other equivalent, as provided by copyright law.

  You may make, run and propagate covered works that you do not
convey, without conditions so long as your license otherwise remains
in force.  You may convey covered works to others for the sole purpose
of having them make modifications exclusively for you, or provide you
with facilities for running those works, provided that you comply with
the terms of this License in conveying all material for which you do
not control copyright.  Those thus making or running the covered works
for you must do so exclusively on your behalf, under your direction
and control, on terms that prohibit them from making any copies of
your copyrighted material outside their relationship with you.

  Conveying under any other circumstances is permitted solely under
the conditions stated below.  Sublicensing is not allowed; section 10
makes it unnecessary.

  3. Protecting Users' Legal Rights From Anti-Circumvention Law.

  No covered work shall be deemed part of an effective technological
measure under any applicable law fulfilling obligations under article
11 of the WIPO copyright treaty adopted on 20 December 1996, or
similar laws prohibiting or restricting circumvention of such
measures.

  When you convey a covered work, you waive any legal power to forbid
circumvention of technological measures to the extent such circumvention
is effected by exercising rights under this License with respect to
the covered work, and you disclaim any intention to limit operation or
modification of the work as a means of enforcing, against the work's
users, your or third parties' legal rights to forbid circumvention of
technological measures.

  4. Conveying Verbatim Copies.

  You may convey verbatim copies of the Program's source code as you
receive it, in any medium, provided that you conspicuously and
appropriately publish on each copy an appropriate copyright notice;
keep intact all notices stating that this License and any
non-permissive terms added in accord with section 7 apply to the code;
keep intact all notices of the absence of any warranty; and give all
recipients a copy of this License along with the Program.

  You may charge any price or no price for each copy that you convey,
and you may offer support or warranty protection for a fee.

  5. Conveying Modified Source Versions.

  You may convey a work based on the Program, or the modifications to
produce it from the Program, in the form of source code under the
terms of section 4, provided that you also meet all of these conditions:

    a) The work must carry prominent notices stating that you modified
    it, and giving a relevant date.

    b) The work must carry prominent notices stating that it is
    released under this License and any conditions added under section
    7.  This requirement modifies the requirement in section 4 to
    "keep intact all notices".

    c) You must license the entire work, as a whole, under this
    License to anyone who comes into possession of a copy.  This
    License will therefore apply, along with any applicable section 7
    additional terms, to the whole of the work, and all its parts,
    regardless of how they are packaged.  This License gives no
    permission to license the work in any other way, but it does not
    invalidate such permission if you have separately received it.

    d) If the work has interactive user interfaces, each must display
    Appropriate Legal Notices; however, if the Program has interactive
    interfaces that do not display Appropriate Legal Notices, your
    work need not make them do so.

  A compilation of a covered work with other separate and independent
works, which are not by their nature extensions of the covered work,
and which are not combined with it such as to form a larger program,
in or on a volume of a storage or distribution medium, is called an
"aggregate" if the compilation and its resulting copyright are not
used to limit the access or legal rights of the compilation's users
beyond what the individual works permit.  Inclusion of a covered work
in an aggregate does not cause this License to apply to the other
parts of the aggregate.

  6. Conveying Non-Source Forms.

  You may convey a covered work in object code form under the terms
of sections 4 and 5, provided that you also convey the
machine-readable Corresponding Source under the terms of this License,
in one of these ways:

    a) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by the
    Corresponding Source fixed on a durable physical medium
    customarily used for software interchange.

    b) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by a
    written offer, valid for at least three years and valid for as
    long as you offer spare parts or customer support for that product
    model, to give anyone who possesses the object code either (1) a
    copy of the Corresponding Source for all the software in the
    product that is covered by this License, on a durable physical
    medium customarily used for software interchange, for a price no
    more than your reasonable cost of physically performing this
    conveying of source, or (2) access to copy the
    Corresponding Source from a network server at no charge.

    c) Convey individual copies of the object code with a copy of the
    written offer to provide the Corresponding Source.  This
    alternative is allowed only occasionally and noncommercially, and
    only if you received the object code with such an offer, in accord
    with subsection 6b.

    d) Convey the object code by offering access from a designated
    place (gratis or for a charge), and offer equivalent access to the
    Corresponding Source in the same way through the same place at no
    further charge.  You need not require recipients to copy the
    Corresponding Source along with the object code.  If the place to
    copy the object code is a network server, the Corresponding Source
    may be on a different server (operated by you or a third party)
    that supports equivalent copying facilities, provided you maintain
    clear directions next to the object code saying where to find the
    Corresponding Source.  Regardless of what server hosts the
    Corresponding Source, you remain obligated to ensure that it is
    available for as long as needed to satisfy these requirements.

    e) Convey the object code using peer-to-peer transmission, provided
    you inform other peers where the object code and Corresponding
    Source of the work are being offered to the general public at no
    charge under subsection 6d.

  A separable portion of the object code, whose source code is excluded
from the Corresponding Source as a System Library, need not be
included in conveying the object code work.

  A "User Product" is either (1) a "consumer product", which means any
tangible personal property which is normally used for personal, family,
or household purposes, or (2) anything designed or sold for incorporation
into a dwelling.  In determining whether a product is a consumer product,
doubtful cases shall be resolved in favor of coverage.  For a particular
product received by a particular user, "normally used" refers to a
typical or common use of that class of product, regardless of the status
of the particular user or of the way in which the particular user
actually uses, or expects or is expected to use, the product.  A product
is a consumer product regardless of whether the product has substantial
commercial, industrial or non-consumer uses, unless such uses represent
the only significant mode of use of the product.

  "Installation Information" for a User Product means any methods,
procedures, authorization keys, or other information required to install
and execute modified versions of a covered work in that User Product from
a modified version of its Corresponding Source.  The information must
suffice to ensure that the continued functioning of the modified object
code is in no case prevented or interfered with solely because
modification has been made.

  If you convey an object code work under this section in, or with, or
specifically for use in, a User Product, and the conveying occurs as
part of a transaction in which the right of possession and use of the
User Product is transferred to the recipient in perpetuity or for a
fixed term (regardless of how the transaction is characterized), the
Corresponding Source conveyed under this section must be accompanied
by the Installation Information.  But this requirement does not apply
if neither you nor any third party retains the ability to install
modified object code on the User Product (for example, the work has
been installed in ROM).

  The requirement to provide Installation Information does not include a
requirement to continue to provide support service, warranty, or updates
for a work that has been modified or installed by the recipient, or for
the User Product in which it has been modified or installed.  Access to a
network may be denied when the modification itself materially and
adversely affects the operation of the network or violates the rules and
protocols for communication across the network.

  Corresponding Source conveyed, and Installation Information provided,
in accord with this section must be in a format that is publicly
documented (and with an implementation available to the public in
source code form), and must require no special password or key for
unpacking, reading or copying.

  7. Additional Terms.

  "Additional permissions" are terms that supplement the terms of this
License by making exceptions from one or more of its conditions.
Additional permissions that are applicable to the entire Program shall
be treated as though they were included in this License, to the extent
that they are valid under applicable law.  If additional permissions
apply only to part of the Program, that part may be used separately
under those permissions, but the entire Program remains governed by
this License without regard to the additional permissions.

  When you convey a copy of a covered work, you may at your option
remove any additional permissions from that copy, or from any part of
it.  (Additional permissions may be written to require their own
removal in certain cases when you modify the work.)  You may place
additional permissions on material, added by you to a covered work,
for which you have or can give appropriate copyright permission.

  Notwithstanding any other provision of this License, for material you
add to a covered work, you may (if authorized by the copyright holders of
that material) supplement the terms of this License with terms:

    a) Disclaiming warranty or limiting liability differently from the
    terms of sections 15 and 16 of this License; or

    b) Requiring preservation of specified reasonable legal notices or
    author attributions in that material or in the Appropriate Legal
    Notices displayed by works containing it; or

    c) Prohibiting misrepresentation of the origin of that material, or
    requiring that modified versions of such material be marked in
    reasonable ways as different from the original version; or

    d) Limiting the use for publicity purposes of names of licensors or
    authors of the material; or

    e) Declining to grant rights under trademark law for use of some
    trade names, trademarks, or service marks; or

    f) Requiring indemnification of licensors and authors of that
    material by anyone who conveys the material (or modified versions of
    it) with contractual assumptions of liability to the recipient, for
    any liability that these contractual assumptions directly impose on
    those licensors and authors.

  All other non-permissive additional terms are considered "further
restrictions" within the meaning of section 10.  If the Program as you
received it, or any part of it, contains a notice stating that it is
governed by this License along with a term that is a further
restriction, you may remove that term.  If a license document contains
a further restriction but permits relicensing or conveying under this
License, you may add to a covered work material governed by the terms
of that license document, provided that the further restriction does
not survive such relicensing or conveying.

  If you add terms to a covered work in accord with this section, you
must place, in the relevant source files, a statement of the
additional terms that apply to those files, or a notice indicating
where to find the applicable terms.

  Additional terms, permissive or non-permissive, may be stated in the
form of a separately written license, or stated as exceptions;
the above requirements apply either way.

  8. Termination.

  You may not propagate or modify a covered work except as expressly
provided under this License.  Any attempt otherwise to propagate or
modify it is void, and will automatically terminate your rights under
this License (including any patent licenses granted under the third
paragraph of section 11).

  However, if you cease all violation of this License, then your
license from a particular copyright holder is reinstated (a)
provisionally, unless and until the copyright holder explicitly and
finally terminates your license, and (b) permanently, if the copyright
holder fails to notify you of the violation by some reasonable means
prior to 60 days after the cessation.

  Moreover, your license from a particular copyright holder is
reinstated permanently if the copyright holder notifies you of the
violation by some reasonable means, this is the first time you have
received notice of violation of this License (for any work) from that
copyright holder, and you cure the violation prior to 30 days after
your receipt of the notice.

  Termination of your rights under this section does not terminate the
licenses of parties who have received copies or rights from you under
this License.  If your rights have been terminated and not permanently
reinstated, you do not qualify to receive new licenses for the same
material under section 10.

  9. Acceptance Not Required for Having Copies.

  You are not required to accept this License in order to receive or
run a copy of the Program.  Ancillary propagation of a covered work
occurring solely as a consequence of using peer-to-peer transmission
to receive a copy likewise does not require acceptance.  However,
nothing other than this License grants you permission to propagate or
modify any covered work.  These actions infringe copyright if you do
not accept this License.  Therefore, by modifying or propagating a
covered work, you indicate your acceptance of this License to do so.

  10. Automatic Licensing of Downstream Recipients.

  Each time you convey a covered work, the recipient automatically
receives a license from the original licensors, to run, modify and
propagate that work, subject to this License.  You are not responsible
for enforcing compliance by third parties with this License.

  An "entity transaction" is a transaction transferring control of an
organization, or substantially all assets of one, or subdividing an
organization, or merging organizations.  If propagation of a covered
work results from an entity transaction, each party to that
transaction who receives a copy of the work also receives whatever
licenses to the work the party's predecessor in interest had or could
give under the previous paragraph, plus a right to possession of the
Corresponding Source of the work from the predecessor in interest, if
the predecessor has it or can get it with reasonable efforts.

  You may not impose any further restrictions on the exercise of the
rights granted or affirmed under this License.  For example, you may
not impose a license fee, royalty, or other charge for exercise of
rights granted under this License, and you may not initiate litigation
(including a cross-claim or counterclaim in a lawsuit) alleging that
any patent claim is infringed by making, using, selling, offering for
sale, or importing the Program or any portion of it.

  11. Patents.

  A "contributor" is a copyright holder who authorizes use under this
License of the Program or a work on which the Program is based.  The
work thus licensed is called the contributor's "contributor version".

  A contributor's "essential patent claims" are all patent claims
owned or controlled by the contributor, whether already acquired or
hereafter acquired, that would be infringed by some manner, permitted
by this License, of making, using, or selling its contributor version,
but do not include claims that would be infringed only as a
consequence of further modification of the contributor version.  For
purposes of this definition, "control" includes the right to grant
patent sublicenses in a manner consistent with the requirements of
this License.

  Each contributor grants you a non-exclusive, worldwide, royalty-free
patent license under the contributor's essential patent claims, to
make, use, sell, offer for sale, import and otherwise run, modify and
propagate the contents of its contributor version.

  In the following three paragraphs, a "patent license" is any express
agreement or commitment, however denominated, not to enforce a patent
(such as an express permission to practice a patent or covenant not to
sue for patent infringement).  To "grant" such a patent license to a
party means to make such an agreement or commitment not to enforce a
patent against the party.

  If you convey a covered work, knowingly relying on a patent license,
and the Corresponding Source of the work is not available for anyone
to copy, free of charge and under the terms of this License, through a
publicly available network server or other readily accessible means,
then you must either (1) cause the Corresponding Source to be so
available, or (2) arrange to deprive yourself of the benefit of the
patent license for this particular work, or (3) arrange, in a manner
consistent with the requirements of this License, to extend the patent
license to downstream recipients.  "Knowingly relying" means you have
actual knowledge that, but for the patent license, your conveying the
covered work in a country, or your recipient's use of the covered work
in a country, would infringe one or more identifiable patents in that
country that you have reason to believe are valid.

  If, pursuant to or in connection with a single transaction or
arrangement, you convey, or propagate by procuring conveyance of, a
covered work, and grant a patent license to some of the parties
receiving the covered work authorizing them to use, propagate, modify
or convey a specific copy of the covered work, then the patent license
you grant is automatically extended to all recipients of the covered
work and works based on it.

  A patent license is "discriminatory" if it does not include within
the scope of its coverage, prohibits the exercise of, or is
conditioned on the non-exercise of one or more of the rights that are
specifically granted under this License.  You may not convey a covered
work if you are a party to an arrangement with a third party that is
in the business of distributing software, under which you make payment
to the third party based on the extent of your activity of conveying
the work, and under which the third party grants, to any of the
parties who would receive the covered work from you, a discriminatory
patent license (a) in connection with copies of the covered work
conveyed by you (or copies made from those copies), or (b) primarily
for and in connection with specific products or compilations that
contain the covered work, unless you entered into that arrangement,
or that patent license was granted, prior to 28 March 2007.

  Nothing in this License shall be construed as excluding or limiting
any implied license or other defenses to infringement that may
otherwise be available to you under applicable patent law.

  12. No Surrender of Others' Freedom.

  If conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot convey a
covered work so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you may
not convey it at all.  For example, if you agree to terms that obligate you
to collect a royalty for further conveying from those to whom you convey
the Program, the only way you could satisfy both those terms and this
License would be to refrain entirely from conveying the Program.

  13. Use with the GNU Affero General Public License.

  Notwithstanding any other provision of this License, you have
permission to link or combine any covered work with a work licensed
under version 3 of the GNU Affero General Public License into a single
combined work, and to convey the resulting work.  The terms of this
License will continue to apply to the part which is the covered work,
but the special requirements of the GNU Affero General Public License,
section 13, concerning interaction through a network will apply to the
combination as such.

  14. Revised Versions of this License.

  The Free Software Foundation may publish revised and/or new versions of
the GNU General Public License from time to time.  Such new versions will
be similar in spirit to the present version, but may differ in detail to
address new problems or concerns.

  Each version is given a distinguishing version number.  If the
Program specifies that a certain numbered version of the GNU General
Public License "or any later version" applies to it, you have the
option of following the terms and conditions either of that numbered
version or of any later version published by the Free Software
Foundation.  If the Program does not specify a version number of the
GNU General Public License, you may choose any version ever published
by the Free Software Foundation.

  If the Program specifies that a proxy can decide which future
versions of the GNU General Public License can be used, that proxy's
public statement of acceptance of a version permanently authorizes you
to choose that version for the Program.

  Later license versions may give you additional or different
permissions.  However, no additional obligations are imposed on any
author or copyright holder as a result of your choosing to follow a
later version.

  15. Disclaimer of Warranty.

  THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY
OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM
IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF
ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

  16. Limitation of Liability.

  IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS
THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY
GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF
DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD
PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),
EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGES.

  17. Interpretation of Sections 15 and 16.

  If the disclaimer of warranty and limitation of liability provided
above cannot be given local legal effect according to their terms,
reviewing courts shall apply local law that most closely approximates
an absolute waiver of all civil liability in connection with the
Program, unless a warranty or assumption of liability accompanies a
copy of the Program in return for a fee.

                     END OF TERMS AND CONDITIONS

            How to Apply These Terms to Your New Programs

  If you develop a new program, and you want it to be of the greatest
possible use to the public, the best way to achieve this is to make it
free software which everyone can redistribute and change under these terms.

  To do so, attach the following notices to the program.  It is safest
to attach them to the start of each source file to most effectively
state the exclusion of warranty; and each file should have at least
the "copyright" line and a pointer to where the full notice is found.

    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) 2021 Manuel Ferreira Junior

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Also add information on how to contact you by electronic and paper mail.

  If the program does terminal interaction, make it output a short
notice like this when it starts in an interactive mode:

    birt-gd  Copyright (C) 2021  Manuel Ferreira Junior
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.

The hypothetical commands `show w' and `show c' should show the appropriate
parts of the General Public License.  Of course, your program's commands
might be different; for a GUI interface, you would use an "about box".

  You should also get your employer (if you work as a programmer) or school,
if any, to sign a "copyright disclaimer" for the program, if necessary.
For more information on this, and how to apply and follow the GNU GPL, see
<https://www.gnu.org/licenses/>.

  The GNU General Public License does not permit incorporating your program
into proprietary programs.  If your program is a subroutine library, you
may consider it more useful to permit linking proprietary applications with
the library.  If this is what you want to do, use the GNU Lesser General
Public License instead of this License.  But first, please read
<https://www.gnu.org/licenses/why-not-lgpl.html>.

# Author
</table>
<table  justify-self="center">
  <tr>
    <td width=5% align="center"><a href="https://manuelfjr.github.io/"><img style="border-radius: 50%;" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/author.png" width="100px;" alt=""/><br /><sub><b>Manuel Ferreira Junior</b></sub></a><br /><a href="https://manuelfjr.github.io/" title=""></a></td>
  </tr> 
</table>

# Contributors

</table>
<table  justify-self="center">
  <tr>
    <td width=5% align="center"><a href="https://github.com/tmfilho"><img style="border-radius: 50%;" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/coauthor1.jpeg" width="100px;" alt=""/><br /><sub><b>Telmo de Menezes e Silva Filho</b></sub></a><br /><a href="https://github.com/tmfilho" title=""></a></td>
    <td width=5% align="center"><a href="https://github.com/jessicareinaldo"><img style="border-radius: 50%;" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/coauthor2.jpeg" width="100px;" alt=""/><br /><sub><b>Jessica Reinaldo</b></sub></a><br /><a href="https://github.com/jessicareinaldo" title=""></a></td>
    <td width=5% align="center"><a href="http://lattes.cnpq.br/2984888073123287"><img style="border-radius: 50%;" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/coauthor3.jpg" width="100px;" alt=""/><br /><sub><b>Ricardo Prudêncio</b></sub></a><br /><a href="http://lattes.cnpq.br/2984888073123287" title=""></a></td>
    <td width=5% align="center"><a href="http://lattes.cnpq.br/5580004940091667"><img style="border-radius: 50%;" src="https://raw.githubusercontent.com/Manuelfjr/birt-gd/main/assets/coauthor4.jpg" width="100px;" alt=""/><br /><sub><b>Eufrásio de Andrade Lima Neto</b></sub></a><br /><a href="http://lattes.cnpq.br/5580004940091667" title=""></a></td>
  </tr> 
</table>


# teste

<div class="plot-container plotly"><div class="user-select-none svg-container" style="position: relative; width: 516px; height: 400px;"><svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="516" height="400" style="background: rgb(255, 255, 255);"><defs id="defs-2bd1a2"><g class="clips"><clipPath id="clip2bd1a2xyplot" class="plotclip"><rect width="316" height="240"></rect></clipPath><clipPath class="axesclip" id="clip2bd1a2x"><rect x="100" y="0" width="316" height="400"></rect></clipPath><clipPath class="axesclip" id="clip2bd1a2y"><rect x="0" y="80" width="516" height="240"></rect></clipPath><clipPath class="axesclip" id="clip2bd1a2xy"><rect x="100" y="80" width="316" height="240"></rect></clipPath></g><g class="gradients"></g></defs><g class="bglayer"><rect class="bg" x="100" y="80" width="316" height="240" style="fill: rgb(175, 175, 175); fill-opacity: 0.2; stroke-width: 0;"></rect></g><g class="draglayer cursor-crosshair"><g class="xy"><rect class="nsewdrag drag" data-subplot="xy" x="100" y="80" width="316" height="240" style="fill: transparent; stroke-width: 0; pointer-events: all;"></rect><rect class="nwdrag drag cursor-nw-resize" data-subplot="xy" x="80" y="60" width="20" height="20" style="fill: transparent; stroke-width: 0; pointer-events: all;"></rect><rect class="nedrag drag cursor-ne-resize" data-subplot="xy" x="416" y="60" width="20" height="20" style="fill: transparent; stroke-width: 0; pointer-events: all;"></rect><rect class="swdrag drag cursor-sw-resize" data-subplot="xy" x="80" y="320" width="20" height="20" style="fill: transparent; stroke-width: 0; pointer-events: all;"></rect><rect class="sedrag drag cursor-se-resize" data-subplot="xy" x="416" y="320" width="20" height="20" style="fill: transparent; stroke-width: 0; pointer-events: all;"></rect><rect class="ewdrag drag cursor-ew-resize" data-subplot="xy" x="131.6" y="321" width="252.8" height="20" style="fill: transparent; stroke-width: 0; pointer-events: all;"></rect><rect class="wdrag drag cursor-w-resize" data-subplot="xy" x="100" y="321" width="31.6" height="20" style="fill: transparent; stroke-width: 0; pointer-events: all;"></rect><rect class="edrag drag cursor-e-resize" data-subplot="xy" x="384.40000000000003" y="321" width="31.6" height="20" style="fill: transparent; stroke-width: 0; pointer-events: all;"></rect><rect class="nsdrag drag cursor-ns-resize" data-subplot="xy" x="79" y="104" width="20" height="192" style="fill: transparent; stroke-width: 0; pointer-events: all;"></rect><rect class="sdrag drag cursor-s-resize" data-subplot="xy" x="79" y="296" width="20" height="24" style="fill: transparent; stroke-width: 0; pointer-events: all;"></rect><rect class="ndrag drag cursor-n-resize" data-subplot="xy" x="79" y="80" width="20" height="24" style="fill: transparent; stroke-width: 0; pointer-events: all;"></rect></g></g><g class="layer-below"><g class="imagelayer"></g><g class="shapelayer"></g></g><g class="cartesianlayer"><g class="subplot xy"><g class="layer-subplot"><g class="shapelayer"></g><g class="imagelayer"></g></g><g class="gridlayer"><g class="x"><path class="xgrid crisp" transform="translate(107.08,0)" d="M0,80v240" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="xgrid crisp" transform="translate(124.53999999999999,0)" d="M0,80v240" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="xgrid crisp" transform="translate(142,0)" d="M0,80v240" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="xgrid crisp" transform="translate(159.46,0)" d="M0,80v240" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="xgrid crisp" transform="translate(176.93,0)" d="M0,80v240" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="xgrid crisp" transform="translate(194.39,0)" d="M0,80v240" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="xgrid crisp" transform="translate(211.85,0)" d="M0,80v240" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="xgrid crisp" transform="translate(229.31,0)" d="M0,80v240" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="xgrid crisp" transform="translate(246.77,0)" d="M0,80v240" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="xgrid crisp" transform="translate(264.24,0)" d="M0,80v240" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="xgrid crisp" transform="translate(281.7,0)" d="M0,80v240" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="xgrid crisp" transform="translate(299.15999999999997,0)" d="M0,80v240" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="xgrid crisp" transform="translate(316.62,0)" d="M0,80v240" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="xgrid crisp" transform="translate(334.09000000000003,0)" d="M0,80v240" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="xgrid crisp" transform="translate(351.55,0)" d="M0,80v240" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="xgrid crisp" transform="translate(369.01,0)" d="M0,80v240" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="xgrid crisp" transform="translate(386.47,0)" d="M0,80v240" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="xgrid crisp" transform="translate(403.93,0)" d="M0,80v240" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path></g><g class="y"><path class="ygrid crisp" transform="translate(0,268.64)" d="M100,0h316" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="ygrid crisp" transform="translate(0,217.27)" d="M100,0h316" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="ygrid crisp" transform="translate(0,165.91)" d="M100,0h316" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path><path class="ygrid crisp" transform="translate(0,114.55)" d="M100,0h316" style="stroke: rgb(255, 255, 255); stroke-opacity: 1; stroke-width: 2px;"></path></g></g><g class="zerolinelayer"></g><path class="xlines-below"></path><path class="ylines-below"></path><g class="overlines-below"></g><g class="xaxislayer-below"></g><g class="yaxislayer-below"></g><g class="overaxes-below"></g><g class="plot" transform="translate(100,80)" clip-path="url('#clip2bd1a2xyplot')"><g class="scatterlayer mlayer"><g class="trace scatter trace32b435" style="stroke-miterlimit: 2; opacity: 1;"><g class="fills"></g><g class="errorbars"></g><g class="lines"><path class="js-line" d="M19.55,15.75L22.04,153.2L24.54,200.04L27.03,216.37L29.53,218.94L32.02,220.38L34.52,208.57L37.01,231.17L39.51,231.47L42,228.29L44.5,233.84L46.99,232.3L49.49,238.46L51.98,232.3L54.47,228.39L56.97,237.02L59.46,234.76L61.96,234.14L64.45,227.57L66.95,235.27L69.44,231.99L71.94,232.6L74.43,220.28L76.93,230.24L79.42,234.25L81.91,226.44L84.41,228.7L86.9,240L89.4,240L91.89,237.43L94.39,227.47L96.88,228.39L99.38,228.49L101.87,222.54L104.37,239.49L106.86,232.4L109.36,236.2L111.85,225.1L114.34,236.1L116.84,228.8L119.33,224.8L121.83,225.1L124.32,239.79L126.82,182.78L129.31,204.56L131.81,216.58L134.3,218.12L136.8,214.52L139.29,138.4L141.79,219.56L144.28,225L146.77,226.13L149.27,224.9L151.76,214.11L154.26,231.17L156.75,217.19L159.25,233.84L161.74,238.87L164.24,233.94L166.73,239.28L169.23,234.04L171.72,236.1L174.21,227.06L176.71,234.25L179.2,234.14L181.7,228.6L184.19,234.14L186.69,210.11L189.18,234.35L191.68,227.67L194.17,234.45L196.67,227.78L199.16,234.45L201.66,219.25L204.15,222.74L206.64,237.12L209.14,234.25L211.63,228.8L214.13,239.59L216.62,227.36L219.12,222.43L221.61,237.84L224.11,231.27L226.6,215.14L229.1,228.19L231.59,239.38L234.09,239.08L236.58,234.45L239.07,217.81L241.57,225.82L244.06,239.79L246.56,228.91L249.05,240L254.04,226.95L256.54,223.36L259.03,223.26L261.53,236.82L264.02,216.58L266.51,234.45L269.01,240L271.5,238.97L274,220.89L276.49,234.14L278.99,223.36L281.48,228.39L283.98,234.04L286.47,229.32L288.97,187.82L291.46,228.6L293.96,232.6L296.45,203.64" style="vector-effect: non-scaling-stroke; fill: none; stroke: rgb(31, 119, 180); stroke-opacity: 1; stroke-width: 2px; opacity: 1;"></path></g><g class="points"><path class="point" transform="translate(19.55,15.75)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(22.04,153.2)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(24.54,200.04)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(27.03,216.37)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(29.53,218.94)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(32.02,220.38)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(34.52,208.57)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(37.01,231.17)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(39.51,231.47)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(42,228.29)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(44.5,233.84)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(46.99,232.3)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(49.49,238.46)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(51.98,232.3)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(54.47,228.39)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(56.97,237.02)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(59.46,234.76)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(61.96,234.14)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(64.45,227.57)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(66.95,235.27)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(69.44,231.99)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(71.94,232.6)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(74.43,220.28)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(76.93,230.24)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(79.42,234.25)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(81.91,226.44)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(84.41,228.7)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(86.9,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(89.4,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(91.89,237.43)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(94.39,227.47)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(96.88,228.39)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(99.38,228.49)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(101.87,222.54)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(104.37,239.49)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(106.86,232.4)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(109.36,236.2)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(111.85,225.1)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(114.34,236.1)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(116.84,228.8)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(119.33,224.8)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(121.83,225.1)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(124.32,239.79)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(126.82,182.78)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(129.31,204.56)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(131.81,216.58)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(134.3,218.12)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(136.8,214.52)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(139.29,138.4)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(141.79,219.56)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(144.28,225)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(146.77,226.13)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(149.27,224.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(151.76,214.11)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(154.26,231.17)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(156.75,217.19)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(159.25,233.84)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(161.74,238.87)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(164.24,233.94)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(166.73,239.28)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(169.23,234.04)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(171.72,236.1)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(174.21,227.06)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(176.71,234.25)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(179.2,234.14)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(181.7,228.6)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(184.19,234.14)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(186.69,210.11)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(189.18,234.35)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(191.68,227.67)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(194.17,234.45)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(196.67,227.78)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(199.16,234.45)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(201.66,219.25)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(204.15,222.74)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(206.64,237.12)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(209.14,234.25)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(211.63,228.8)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(214.13,239.59)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(216.62,227.36)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(219.12,222.43)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(221.61,237.84)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(224.11,231.27)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(226.6,215.14)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(229.1,228.19)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(231.59,239.38)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(234.09,239.08)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(236.58,234.45)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(239.07,217.81)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(241.57,225.82)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(244.06,239.79)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(246.56,228.91)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(249.05,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(251.55,233.84)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(254.04,226.95)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(256.54,223.36)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(259.03,223.26)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(261.53,236.82)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(264.02,216.58)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(266.51,234.45)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(269.01,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(271.5,238.97)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(274,220.89)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(276.49,234.14)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(278.99,223.36)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(281.48,228.39)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(283.98,234.04)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(286.47,229.32)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(288.97,187.82)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(291.46,228.6)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(293.96,232.6)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(296.45,203.64)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path></g><g class="text"></g></g><g class="trace scatter tracea10664" style="stroke-miterlimit: 2; opacity: 1;"><g class="fills"></g><g class="errorbars"></g><g class="lines"><path class="js-line" d="M19.55,185.97L22.04,227.26L24.54,238.97L27.03,238.97L29.53,239.9L32.02,234.56L34.52,231.78L37.01,239.59L39.51,239.9L54.47,239.9L56.97,238.05L59.46,235.38L61.96,238.46L64.45,238.66L66.95,239.79L74.43,240L76.93,237.74L79.42,240L81.91,240L104.37,239.9L106.86,232.4L109.36,239.9L111.85,239.9L124.32,240L126.82,228.29L129.31,236.51L131.81,239.38L134.3,236.61L136.8,236.51L139.29,219.76L141.79,238.97L146.77,234.56L149.27,238.05L154.26,239.9L156.75,239.59L159.25,239.9L161.74,238.97L164.24,234.35L166.73,239.28L169.23,239.59L171.72,238.56L176.71,239.79L179.2,239.69L181.7,234.45L184.19,239.9L201.66,240L204.15,239.9L206.64,237.23L209.14,240L221.61,239.69L224.11,238.77L229.1,239.38L231.59,239.49L236.58,240L239.07,240L244.06,239.9L246.56,228.91L249.05,240L251.55,240L261.53,240L264.02,238.97L266.51,240L269.01,240L283.98,240L286.47,236.82L288.97,221.3L291.46,233.12L293.96,239.79L296.45,233.84" style="vector-effect: non-scaling-stroke; fill: none; stroke: rgb(255, 127, 14); stroke-opacity: 1; stroke-width: 2px; opacity: 1;"></path></g><g class="points"><path class="point" transform="translate(19.55,185.97)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(22.04,227.26)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(24.54,238.97)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(27.03,238.97)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(29.53,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(32.02,234.56)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(34.52,231.78)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(37.01,239.59)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(39.51,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(42,239.79)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(44.5,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(46.99,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(49.49,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(51.98,239.79)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(54.47,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(56.97,238.05)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(59.46,235.38)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(61.96,238.46)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(64.45,238.66)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(66.95,239.79)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(69.44,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(71.94,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(74.43,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(76.93,237.74)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(79.42,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(81.91,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(84.41,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(86.9,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(89.4,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(91.89,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(94.39,239.69)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(96.88,239.49)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(99.38,239.59)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(101.87,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(104.37,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(106.86,232.4)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(109.36,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(111.85,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(114.34,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(116.84,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(119.33,239.79)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(121.83,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(124.32,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(126.82,228.29)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(129.31,236.51)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(131.81,239.38)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(134.3,236.61)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(136.8,236.51)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(139.29,219.76)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(141.79,238.97)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(144.28,236.71)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(146.77,234.56)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(149.27,238.05)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(151.76,238.97)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(154.26,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(156.75,239.59)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(159.25,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(161.74,238.97)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(164.24,234.35)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(166.73,239.28)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(169.23,239.59)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(171.72,238.56)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(174.21,239.28)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(176.71,239.79)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(179.2,239.69)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(181.7,234.45)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(184.19,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(186.69,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(189.18,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(191.68,239.69)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(194.17,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(196.67,239.69)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(199.16,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(201.66,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(204.15,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(206.64,237.23)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(209.14,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(211.63,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(214.13,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(216.62,239.59)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(219.12,239.79)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(221.61,239.69)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(224.11,238.77)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(226.6,239.38)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(229.1,239.38)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(231.59,239.49)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(234.09,239.38)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(236.58,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(239.07,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(241.57,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(244.06,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(246.56,228.91)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(249.05,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(251.55,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(254.04,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(256.54,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(259.03,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(261.53,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(264.02,238.97)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(266.51,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(269.01,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(271.5,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(274,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(276.49,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(278.99,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(281.48,239.9)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(283.98,240)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(286.47,236.82)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(288.97,221.3)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(291.46,233.12)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(293.96,239.79)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path><path class="point" transform="translate(296.45,233.84)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path></g><g class="text"></g></g></g></g><g class="overplot"></g><path class="xlines-above crisp" d="M98,321H416" style="fill: none; stroke-width: 2px; stroke: rgb(148, 148, 148); stroke-opacity: 1;"></path><path class="ylines-above crisp" d="M99,80V320" style="fill: none; stroke-width: 2px; stroke: rgb(148, 148, 148); stroke-opacity: 1;"></path><g class="overlines-above"></g><g class="xaxislayer-above"><g class="xtick"><text text-anchor="end" x="0" y="336.4" data-unformatted="10-25" data-math="N" transform="translate(107.08,0) rotate(-45,0,330.4)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">10-25</text></g><g class="xtick"><text text-anchor="end" x="0" y="336.4" data-unformatted="11-01" data-math="N" transform="translate(124.53999999999999,0) rotate(-45,0,330.4)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">11-01</text></g><g class="xtick"><text text-anchor="end" x="0" y="336.4" data-unformatted="11-08" data-math="N" transform="translate(142,0) rotate(-45,0,330.4)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">11-08</text></g><g class="xtick"><text text-anchor="end" x="0" y="336.4" data-unformatted="11-15" data-math="N" transform="translate(159.46,0) rotate(-45,0,330.4)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">11-15</text></g><g class="xtick"><text text-anchor="end" x="0" y="336.4" data-unformatted="11-22" data-math="N" transform="translate(176.93,0) rotate(-45,0,330.4)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">11-22</text></g><g class="xtick"><text text-anchor="end" x="0" y="336.4" data-unformatted="11-29" data-math="N" transform="translate(194.39,0) rotate(-45,0,330.4)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">11-29</text></g><g class="xtick"><text text-anchor="end" x="0" y="336.4" data-unformatted="12-06" data-math="N" transform="translate(211.85,0) rotate(-45,0,330.4)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">12-06</text></g><g class="xtick"><text text-anchor="end" x="0" y="336.4" data-unformatted="12-13" data-math="N" transform="translate(229.31,0) rotate(-45,0,330.4)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">12-13</text></g><g class="xtick"><text text-anchor="end" x="0" y="336.4" data-unformatted="12-20" data-math="N" transform="translate(246.77,0) rotate(-45,0,330.4)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">12-20</text></g><g class="xtick"><text text-anchor="end" x="0" y="336.4" data-unformatted="12-27" data-math="N" transform="translate(264.24,0) rotate(-45,0,330.4)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">12-27</text></g><g class="xtick"><text text-anchor="end" x="0" y="336.4" data-unformatted="01-03" data-math="N" transform="translate(281.7,0) rotate(-45,0,330.4)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">01-03</text></g><g class="xtick"><text text-anchor="end" x="0" y="336.4" data-unformatted="01-10" data-math="N" transform="translate(299.15999999999997,0) rotate(-45,0,330.4)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">01-10</text></g><g class="xtick"><text text-anchor="end" x="0" y="336.4" data-unformatted="01-17" data-math="N" transform="translate(316.62,0) rotate(-45,0,330.4)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">01-17</text></g><g class="xtick"><text text-anchor="end" x="0" y="336.4" data-unformatted="01-24" data-math="N" transform="translate(334.09000000000003,0) rotate(-45,0,330.4)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">01-24</text></g><g class="xtick"><text text-anchor="end" x="0" y="336.4" data-unformatted="01-31" data-math="N" transform="translate(351.55,0) rotate(-45,0,330.4)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">01-31</text></g><g class="xtick"><text text-anchor="end" x="0" y="336.4" data-unformatted="02-07" data-math="N" transform="translate(369.01,0) rotate(-45,0,330.4)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">02-07</text></g><g class="xtick"><text text-anchor="end" x="0" y="336.4" data-unformatted="02-14" data-math="N" transform="translate(386.47,0) rotate(-45,0,330.4)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">02-14</text></g><g class="xtick"><text text-anchor="end" x="0" y="336.4" data-unformatted="02-21" data-math="N" transform="translate(403.93,0) rotate(-45,0,330.4)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">02-21</text></g></g><g class="yaxislayer-above"><g class="ytick"><text text-anchor="end" x="95.6" y="4.199999999999999" data-unformatted="0" data-math="N" transform="translate(0,320)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">0</text></g><g class="ytick"><text text-anchor="end" x="95.6" y="4.199999999999999" data-unformatted="500" data-math="N" transform="translate(0,268.64)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">500</text></g><g class="ytick"><text text-anchor="end" x="95.6" y="4.199999999999999" data-unformatted="1,000" data-math="N" transform="translate(0,217.27)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">1,000</text></g><g class="ytick"><text text-anchor="end" x="95.6" y="4.199999999999999" data-unformatted="1,500" data-math="N" transform="translate(0,165.91)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">1,500</text></g><g class="ytick"><text text-anchor="end" x="95.6" y="4.199999999999999" data-unformatted="2,000" data-math="N" transform="translate(0,114.55)" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">2,000</text></g></g><g class="overaxes-above"></g></g></g><g class="polarlayer"></g><g class="ternarylayer"></g><g class="geolayer"></g><g class="funnelarealayer"></g><g class="pielayer"></g><g class="treemaplayer"></g><g class="sunburstlayer"></g><g class="glimages"></g></svg><div class="gl-container"></div><svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="516" height="400"><defs id="topdefs-2bd1a2"><g class="clips"></g><clipPath id="legend2bd1a2"><rect width="143" height="48" x="0" y="0"></rect></clipPath></defs><g class="indicatorlayer"></g><g class="layer-above"><g class="imagelayer"></g><g class="shapelayer"></g></g><g class="infolayer"><g class="legend" pointer-events="all" transform="translate(100,176)"><rect class="bg" shape-rendering="crispEdges" style="stroke: rgb(68, 68, 68); stroke-opacity: 1; fill: rgb(231, 231, 231); fill-opacity: 1; stroke-width: 0px;" width="143" height="48" x="0" y="0"></rect><g class="scrollbox" transform="" clip-path="url('#legend2bd1a2')"><g class="groups"><g class="traces" transform="translate(0,14.5)" style="opacity: 1;"><text class="legendtext" text-anchor="start" x="40" y="4.680000000000001" data-unformatted="With_Mirrors" data-math="N" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">With_Mirrors</text><g class="layers" style="opacity: 1;"><g class="legendfill"></g><g class="legendlines"><path class="js-line" d="M5,0h30" style="fill: none; stroke: rgb(31, 119, 180); stroke-opacity: 1; stroke-width: 2px;"></path></g><g class="legendsymbols"><g class="legendpoints"><path class="scatterpts" transform="translate(20,0)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(31, 119, 180); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path></g></g></g><rect class="legendtoggle" pointer-events="all" x="0" y="-9.5" width="137.767578125" height="19" style="cursor: pointer; fill: rgb(0, 0, 0); fill-opacity: 0;"></rect></g><g class="traces" transform="translate(0,33.5)" style="opacity: 1;"><text class="legendtext" text-anchor="start" x="40" y="4.680000000000001" data-unformatted="Without_Mirrors" data-math="N" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">Without_Mirrors</text><g class="layers" style="opacity: 1;"><g class="legendfill"></g><g class="legendlines"><path class="js-line" d="M5,0h30" style="fill: none; stroke: rgb(255, 127, 14); stroke-opacity: 1; stroke-width: 2px;"></path></g><g class="legendsymbols"><g class="legendpoints"><path class="scatterpts" transform="translate(20,0)" d="M3,0A3,3 0 1,1 0,-3A3,3 0 0,1 3,0Z" style="opacity: 1; stroke-width: 1px; fill: rgb(255, 127, 14); fill-opacity: 1; stroke: rgb(68, 68, 68); stroke-opacity: 1;"></path></g></g></g><rect class="legendtoggle" pointer-events="all" x="0" y="-9.5" width="137.767578125" height="19" style="cursor: pointer; fill: rgb(0, 0, 0); fill-opacity: 0;"></rect></g></g></g><rect class="scrollbar" rx="20" ry="3" width="0" height="0" style="fill: rgb(128, 139, 164); fill-opacity: 1;" x="0" y="0"></rect></g><g class="rangeselector" transform="translate(100,56)" style="cursor: pointer; pointer-events: all;"><g class="button" transform=""><rect class="selector-rect" shape-rendering="crispEdges" rx="3" ry="3" style="stroke: rgb(68, 68, 68); stroke-opacity: 1; fill: rgb(238, 238, 238); fill-opacity: 1; stroke-width: 0px;" x="0" y="0" width="32.734375" height="19"></rect><text class="selector-text" text-anchor="middle" data-unformatted="30d" data-math="N" x="16.3671875" y="12.5" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">30d</text></g><g class="button" transform="translate(37.734375,0)"><rect class="selector-rect" shape-rendering="crispEdges" rx="3" ry="3" style="stroke: rgb(68, 68, 68); stroke-opacity: 1; fill: rgb(238, 238, 238); fill-opacity: 1; stroke-width: 0px;" x="0" y="0" width="32.734375" height="19"></rect><text class="selector-text" text-anchor="middle" data-unformatted="60d" data-math="N" x="16.3671875" y="12.5" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">60d</text></g><g class="button" transform="translate(75.46875,0)"><rect class="selector-rect" shape-rendering="crispEdges" rx="3" ry="3" style="stroke: rgb(68, 68, 68); stroke-opacity: 1; fill: rgb(238, 238, 238); fill-opacity: 1; stroke-width: 0px;" x="0" y="0" width="32.734375" height="19"></rect><text class="selector-text" text-anchor="middle" data-unformatted="90d" data-math="N" x="16.3671875" y="12.5" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">90d</text></g><g class="button" transform="translate(113.203125,0)"><rect class="selector-rect" shape-rendering="crispEdges" rx="3" ry="3" style="stroke: rgb(68, 68, 68); stroke-opacity: 1; fill: rgb(238, 238, 238); fill-opacity: 1; stroke-width: 0px;" x="0" y="0" width="30" height="19"></rect><text class="selector-text" text-anchor="middle" data-unformatted="all" data-math="N" x="15" y="12.5" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;">all</text></g></g><g class="g-gtitle"><text class="gtitle" x="258" y="40" text-anchor="middle" dy="0em" data-unformatted="Daily Download Quantity of birt-gd package - Overall" data-math="N" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 17px; fill: rgb(68, 68, 68); opacity: 1; font-weight: normal; white-space: pre;">Daily Download Quantity of birt-gd package - Overall</text></g><g class="g-xtitle" transform="translate(0,1.197265625)"><text class="xtitle" x="258" y="379" text-anchor="middle" data-unformatted="Date" data-math="N" style="font-family: Geneva, Verdana, Geneva, sans-serif; font-size: 16px; fill: rgb(127, 127, 127); opacity: 1; font-weight: normal; white-space: pre;">Date</text></g><g class="g-ytitle" transform="translate(-5.291015625,0)"><text class="ytitle" transform="rotate(-90,61,200)" x="61" y="200" text-anchor="middle" data-unformatted="Downloads" data-math="N" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 14px; fill: rgb(68, 68, 68); opacity: 1; font-weight: normal; white-space: pre;">Downloads</text></g></g><g class="menulayer"></g><g class="zoomlayer"></g></svg><div class="modebar-container" style="position: absolute; top: 0px; right: 0px; width: 100%;"><div id="modebar-2bd1a2" class="modebar modebar--hover ease-bg"><div class="modebar-group"></div><div class="modebar-group"></div><div class="modebar-group"><a rel="tooltip" class="modebar-btn" data-title="Autoscale" data-attr="zoom" data-val="auto" data-toggle="false" data-gravity="n"><svg viewBox="0 0 1000 1000" class="icon" height="1em" width="1em"><path d="m250 850l-187 0-63 0 0-62 0-188 63 0 0 188 187 0 0 62z m688 0l-188 0 0-62 188 0 0-188 62 0 0 188 0 62-62 0z m-875-938l0 188-63 0 0-188 0-62 63 0 187 0 0 62-187 0z m875 188l0-188-188 0 0-62 188 0 62 0 0 62 0 188-62 0z m-125 188l-1 0-93-94-156 156 156 156 92-93 2 0 0 250-250 0 0-2 93-92-156-156-156 156 94 92 0 2-250 0 0-250 0 0 93 93 157-156-157-156-93 94 0 0 0-250 250 0 0 0-94 93 156 157 156-157-93-93 0 0 250 0 0 250z" transform="matrix(1 0 0 -1 0 850)"></path></svg></a><a rel="tooltip" class="modebar-btn" data-title="Reset axes" data-attr="zoom" data-val="reset" data-toggle="false" data-gravity="n"><svg viewBox="0 0 928.6 1000" class="icon" height="1em" width="1em"><path d="m786 296v-267q0-15-11-26t-25-10h-214v214h-143v-214h-214q-15 0-25 10t-11 26v267q0 1 0 2t0 2l321 264 321-264q1-1 1-4z m124 39l-34-41q-5-5-12-6h-2q-7 0-12 3l-386 322-386-322q-7-4-13-4-7 2-12 7l-35 41q-4 5-3 13t6 12l401 334q18 15 42 15t43-15l136-114v109q0 8 5 13t13 5h107q8 0 13-5t5-13v-227l122-102q5-5 6-12t-4-13z" transform="matrix(1 0 0 -1 0 850)"></path></svg></a></div><div class="modebar-group"><a rel="tooltip" class="modebar-btn" data-title="Show closest data on hover" data-attr="hovermode" data-val="closest" data-toggle="false" data-gravity="ne"><svg viewBox="0 0 1500 1000" class="icon" height="1em" width="1em"><path d="m375 725l0 0-375-375 375-374 0-1 1125 0 0 750-1125 0z" transform="matrix(1 0 0 -1 0 850)"></path></svg></a><a rel="tooltip" class="modebar-btn active" data-title="Compare data on hover" data-attr="hovermode" data-val="x" data-toggle="false" data-gravity="ne"><svg viewBox="0 0 1125 1000" class="icon" height="1em" width="1em"><path d="m187 786l0 2-187-188 188-187 0 0 937 0 0 373-938 0z m0-499l0 1-187-188 188-188 0 0 937 0 0 376-938-1z" transform="matrix(1 0 0 -1 0 850)"></path></svg></a></div></div></div><svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="516" height="400"><g class="hoverlayer"></g></svg></div></div>